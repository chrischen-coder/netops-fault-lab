"""Self-contained HTML; report fields are escaped and no scripts are loaded."""

from collections import deque
from html import escape

from .schema import Case


def topology_svg(case: Case, report: dict) -> str:
    nodes = sorted({endpoint for link in case.links for endpoint in (link.source, link.target)})
    neighbors = {node: set() for node in nodes}
    for link in case.links:
        neighbors[link.source].add(link.target)
        neighbors[link.target].add(link.source)
    levels = {}
    while len(levels) < len(nodes):
        start = min((n for n in nodes if n not in levels), key=lambda n: (-len(neighbors[n]), n))
        levels[start] = max(levels.values(), default=-2) + 2
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(neighbors[current]):
                if neighbor not in levels:
                    levels[neighbor] = levels[current] + 1
                    queue.append(neighbor)
    points = {}
    for level in sorted(set(levels.values())):
        layer = sorted(n for n in nodes if levels[n] == level)
        for i, node in enumerate(layer):
            points[node] = (60 + 440 * level / max(levels.values(), default=1),
                            175 if len(layer) == 1 else 50 + 250 * i / (len(layer) - 1))
    highlighted = set(report['indistinguishable_from_top'])
    color = '#bd6a0a' if report['status'] == 'ambiguous' else '#167667'
    parts = ['<svg viewBox="0 0 560 360" role="img" aria-label="Network links; leading candidates highlighted">']
    for link in case.links:
        x1, y1 = points[link.source]
        x2, y2 = points[link.target]
        stroke = color if link.id in highlighted else '#ced9dd'
        label_color = color if link.id in highlighted else '#55717d'
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="4"/>')
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        label = escape(link.id[:20])
        parts.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" fill="{label_color}" font-size="12" font-weight="700" stroke="white" stroke-width="4" paint-order="stroke">{label}</text>')
    for node, (x, y) in points.items():
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#173f4b"/>')
        parts.append(f'<text x="{x:.1f}" y="{y + 24:.1f}" text-anchor="middle" font-size="13" fill="#173f4b">{escape(node[:22])}</text>')
    return ''.join(parts) + '</svg>'


def render_text(report: dict) -> str:
    lines = [f"NetOps Fault Lab | {report['case_id']}", f"Status: {report['status']}",
             f"Observed: {report['observed_probes']} | Missing: {report['missing_probes']}",
             f"Model: {report['model']['provenance'].get('source', 'unspecified')}",
             "Hypothesis          Model probability"]
    for item in report['ranking'][:5]:
        lines.append(f"{item['hypothesis']:<20} {item['probability']:.3f}")
    if len(report['indistinguishable_from_top']) > 1:
        lines.append('Indistinguishable: ' + ', '.join(report['indistinguishable_from_top']))
    if report['next_probes']:
        probe = report['next_probes'][0]
        lines.append(f"Next probe: {probe['id']} ({' -> '.join(probe['path'])}); expected gain {probe['expected_gain_bits']:.3f} bits")
    else:
        lines.append('No informative probe in the supplied candidate list.')
    lines.append('Probabilities assume at most one failed link and independent observations.')
    return '\n'.join(lines)


def render_html(case: Case, report: dict) -> str:
    e = lambda value: escape(str(value), quote=True)
    status = report['status']
    titles = {
        'ambiguous': 'These links fit the same observations.',
        'insufficient_evidence': 'No observed outcomes to diagnose.',
        'review': 'The evidence does not support a clear choice.',
        'candidate': 'A leading candidate to investigate.',
        'consistent_with_healthy': 'The observations are consistent with a healthy network.',
    }
    lead = report['ranking'][0]
    rows = ''.join(f'<tr><td>{e(item["hypothesis"])}</td><td><div class="bar" style="width:{100 * item["probability"]:.3f}%"></div></td><td class="number">{100 * item["probability"]:.1f}%</td></tr>'
                   for item in report['ranking'])
    probe = report['next_probes'][0] if report['next_probes'] else None
    next_action = (f'<div class="eyebrow">NEXT MEASUREMENT</div><h2>{e(probe["id"])}</h2>'
                   f'<p class="route">{e(" → ".join(probe["path"]))}</p>'
                   f'<p><strong>{probe["expected_gain_bits"]:.3f} bits</strong> expected information gain · cost {probe["cost"]:g}</p>'
                   '<p class="muted">Selected from the paths you supplied. No network request has been sent.</p>') if probe else (
                   '<div class="eyebrow">NEXT MEASUREMENT</div><h2>No useful candidate supplied</h2><p>Add a probe whose path separates the remaining hypotheses.</p>')
    evidence_rows = ''.join(f'<tr><td>{e(p["id"])}</td><td><span class="{e(p["outcome"])}">{e(p["outcome"])}</span></td><td>{e(" → ".join(p["path"]))}</td><td class="number">{p["supports_top_vs_runner_up_log_bayes_factor"]:+.3f}</td></tr>'
                            for p in report['evidence_comparison']['probes'])
    group = ', '.join(report['indistinguishable_from_top'])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>NetOps Fault Lab — {e(case.id)}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f6;color:#17313b;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1160px;padding:32px 28px 48px;margin:auto}}header{{display:flex;justify-content:space-between;gap:16px;padding-bottom:24px;border-bottom:1px solid #d8e2e5}}.brand{{font-weight:750;font-size:20px}}.muted{{color:#5c747e;font-size:13px}}.hero{{padding:28px 0 18px}}.eyebrow{{font-size:11px;letter-spacing:1.8px;font-weight:750;color:#39706d}}h1{{font-size:32px;line-height:1.2;letter-spacing:-.6px;margin:10px 0}}h2{{font-size:20px;margin:8px 0}}p{{margin:10px 0}}.pill{{display:inline-block;padding:5px 10px;border-radius:5px;background:#fff3dd;color:#8c5410;font-size:12px}}.grid{{display:grid;grid-template-columns:1.05fr 1fr;gap:20px}}.card{{background:white;border:1px solid #dce5e8;border-radius:12px;padding:22px;min-width:0}}.action{{background:#e8f3ef;border-color:#bfdad0;margin-top:20px}}svg{{width:100%;display:block}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px 6px;border-bottom:1px solid #edf1f3;text-align:left;overflow-wrap:anywhere}}th{{color:#607780;font-weight:500}}.number{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}.bar{{height:9px;background:#ce913c;min-width:1px;border-radius:6px}}td:nth-child(2){{min-width:90px}}.route{{font-size:22px;font-weight:600;overflow-wrap:anywhere}}.fail{{color:#b44f42}}.pass{{color:#167667}}.evidence{{margin-top:20px;overflow:auto}}.note{{margin-top:20px;padding-top:16px;border-top:1px solid #d8e2e5;font-size:12px;color:#5c747e}}details{{margin-top:16px}}summary{{cursor:pointer;font-weight:600}}code{{overflow-wrap:anywhere}}@media(max-width:760px){{main{{padding:20px 14px}}.grid{{grid-template-columns:1fr}}h1{{font-size:26px}}header{{flex-direction:column}}.card{{padding:16px}}}}
</style></head><body><main>
<header><div class="brand">NetOps Fault Lab</div><div class="muted">Offline diagnostic report · {e(case.id)}</div></header>
<section class="hero"><div class="eyebrow">PATH OBSERVATIONS → FAULT HYPOTHESES</div><h1>{titles[status]}</h1><p><span class="pill">{e(status.replace('_',' '))}</span> &nbsp; {report['observed_probes']} observed probes · {report['missing_probes']} missing</p></section>
<div class="grid"><section class="card"><div class="eyebrow">TOPOLOGY</div>{topology_svg(case,report)}<p class="muted">Highlighted candidates: {e(group)}</p></section>
<section class="card"><div class="eyebrow">CANDIDATE RANKING</div><h2>Probability under the model</h2><table><thead><tr><th>Hypothesis</th><th></th><th class="number">Posterior</th></tr></thead><tbody>{rows}</tbody></table><p class="muted">Equal path signatures cannot be separated by the current observations, even when the prior favors one.</p></section></div>
<section class="card action">{next_action}</section>
<section class="card evidence"><div class="eyebrow">WHY THE RANKING LOOKS THIS WAY</div><p>Evidence for <strong>{e(lead['hypothesis'])}</strong> versus <strong>{e(report['evidence_comparison']['runner_up'])}</strong>. Zero means the probe does not distinguish them.</p><table><thead><tr><th>Probe</th><th>Result</th><th>Path</th><th>Log Bayes factor</th></tr></thead><tbody>{evidence_rows}</tbody></table>
<details><summary>Model assumptions and parameters</summary><p>At most one failed link, known static paths and conditionally independent outcomes. Repeated copies of one alarm are not independent evidence. Rankings do not establish causation.</p><p>Failure on an affected path: {report['model']['hit_failure_probability']:.4f}; background failure: {report['model']['background_failure_probability']:.4f}; healthy prior: {report['model']['healthy_prior']:.4f}.</p></details></section>
<footer class="note">Generated by NetOps Fault Lab {e(report['tool_version'])}. Synthetic examples demonstrate the method; they do not establish production accuracy. Review paths and assumptions before acting.</footer>
</main></body></html>'''
