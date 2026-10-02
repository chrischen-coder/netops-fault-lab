"""Strict input records. Ground truth is deliberately separate from a Case."""

from dataclasses import asdict, dataclass, field
import gzip
import json
import math
from pathlib import Path

HEALTHY = "__healthy__"


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{name} must be a nonempty string of at most 200 characters")
    if any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError(f"{name} contains invalid control or Unicode characters")
    return value


def _object(value: object, required: set[str], optional: set[str], name: str) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError(f"Invalid {name} fields; required={sorted(required)}, optional={sorted(optional)}")
    return value


def _list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class Link:
    id: str
    source: str
    target: str


@dataclass(frozen=True)
class Probe:
    id: str
    path: tuple[str, ...]
    links: frozenset[str]
    outcome: str | None = None
    cost: float = 1.0


@dataclass(frozen=True)
class Case:
    id: str
    topology_id: str
    links: tuple[Link, ...]
    observations: tuple[Probe, ...]
    available_probes: tuple[Probe, ...]

    @classmethod
    def from_dict(cls, value: object) -> "Case":
        raw = _object(value, {"id", "topology_id", "links", "observations"},
                      {"available_probes", "schema_version"}, "case (truth is not an input)")
        if raw.get("schema_version", 1) != 1 or isinstance(raw.get("schema_version", 1), bool):
            raise ValueError("Unsupported case schema_version")
        links, endpoints, identifiers = [], {}, set()
        for item in _list(raw["links"], "links"):
            item = _object(item, {"id", "source", "target"}, set(), "link")
            identifier = _identifier(item["id"], "link id")
            source, target = (_identifier(item[k], k) for k in ("source", "target"))
            edge = frozenset((source, target))
            if identifier in identifiers or identifier == HEALTHY or edge in endpoints or source == target:
                raise ValueError("Links need unique IDs and endpoints; self/parallel links are unsupported")
            identifiers.add(identifier)
            endpoints[edge] = identifier
            links.append(Link(identifier, source, target))
        if not links:
            raise ValueError("At least one link is required")
        probe_ids: set[str] = set()

        def parse_probe(item: object, observed: bool) -> Probe:
            required = {"id", "path", "outcome"} if observed else {"id", "path"}
            item = _object(item, required, set() if observed else {"cost"}, "probe")
            identifier = _identifier(item["id"], "probe id")
            if identifier in probe_ids:
                raise ValueError("Probe IDs must be unique across observations and available probes")
            probe_ids.add(identifier)
            path = tuple(_identifier(v, "path node") for v in _list(item["path"], "path"))
            if len(path) < 2 or len(path) != len(set(path)):
                raise ValueError("A probe path needs at least two nodes without repeated nodes")
            try:
                route = frozenset(endpoints[frozenset(pair)] for pair in zip(path, path[1:]))
            except KeyError as error:
                raise ValueError("Probe path contains an edge absent from the topology") from error
            outcome = item.get("outcome")
            if observed and outcome not in ("pass", "fail", "missing"):
                raise ValueError("outcome must be pass, fail or missing")
            cost = _number(item.get("cost", 1.0), "probe cost")
            if cost <= 0:
                raise ValueError("Probe cost must be positive")
            return Probe(identifier, path, route, outcome, cost)

        observations = tuple(parse_probe(p, True) for p in _list(raw["observations"], "observations"))
        available = tuple(parse_probe(p, False) for p in _list(raw.get("available_probes", []), "available_probes"))
        return cls(_identifier(raw["id"], "case id"), _identifier(raw["topology_id"], "topology_id"),
                   tuple(links), observations, available)

    @property
    def hypotheses(self) -> tuple[str, ...]:
        return (HEALTHY, *sorted(link.id for link in self.links))

    @property
    def observed(self) -> tuple[Probe, ...]:
        return tuple(p for p in self.observations if p.outcome != "missing")


@dataclass(frozen=True)
class NoiseModel:
    hit_failure_probability: float = 0.9
    background_failure_probability: float = 0.03
    healthy_prior: float = 0.15
    provenance: dict = field(default_factory=lambda: {"source": "illustrative defaults; not calibrated"})

    def __post_init__(self) -> None:
        hit = _number(self.hit_failure_probability, "hit_failure_probability")
        background = _number(self.background_failure_probability, "background_failure_probability")
        prior = _number(self.healthy_prior, "healthy_prior")
        if not 0 < background < hit < 1 or not 0 < prior < 1:
            raise ValueError("Require 0 < background < hit < 1 and 0 < healthy_prior < 1")
        if not isinstance(self.provenance, dict):
            raise ValueError("Model provenance must be an object")

    def to_dict(self) -> dict:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value: object) -> "NoiseModel":
        raw = _object(value, {"schema_version", "hit_failure_probability", "background_failure_probability",
                              "healthy_prior"}, {"provenance"}, "model")
        if raw["schema_version"] != 1 or isinstance(raw["schema_version"], bool):
            raise ValueError("Unsupported model schema_version")
        return cls(**{k: v for k, v in raw.items() if k != "schema_version"})


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_json(text: str) -> object:
    return json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)


def read_json(path: str | Path) -> object:
    return parse_json(Path(path).read_text(encoding="utf-8"))


def read_dataset(path: str | Path) -> list[tuple[Case, str]]:
    result = []
    ids = set()
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            content = stream.read()
    else:
        content = path.read_text(encoding='utf-8')
    for line, text in enumerate(content.splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = _object(parse_json(text), {"case", "truth"}, set(), "labeled row")
            case = Case.from_dict(row["case"])
            truth = _identifier(row["truth"], "truth")
            if truth not in case.hypotheses:
                raise ValueError("Truth must be a link ID or __healthy__")
            if case.id in ids:
                raise ValueError("Case IDs must be unique")
            ids.add(case.id)
            result.append((case, truth))
        except ValueError as error:
            raise ValueError(f"Invalid dataset row {line}: {error}") from error
    if not result:
        raise ValueError("Dataset is empty")
    return result
