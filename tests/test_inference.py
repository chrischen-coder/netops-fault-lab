import copy
from dataclasses import replace
import math
from pathlib import Path
import unittest

from netops_fault_lab import Case, NoiseModel, diagnose, fit_noise
from netops_fault_lab.inference import entropy, posterior, probe_information
from netops_fault_lab.schema import HEALTHY, read_json

ROOT = Path(__file__).resolve().parents[1]


def minimal(outcome='fail'):
    return {'id':'tiny', 'topology_id':'single-edge',
            'links':[{'id':'edge','source':'a','target':'b'}],
            'observations':[{'id':'p','path':['a','b'],'outcome':outcome}],
            'available_probes':[{'id':'next','path':['a','b'],'cost':1}]}


class InferenceTests(unittest.TestCase):
    def test_one_link_matches_bayes_by_hand(self):
        model = NoiseModel(.8,.1,.2)
        values = posterior(Case.from_dict(minimal()), model)
        self.assertAlmostEqual(values['edge'], .8*.8/(.8*.8+.2*.1))
        self.assertAlmostEqual(sum(values.values()), 1)

    def test_pass_is_negative_evidence_for_an_on_path_fault(self):
        model = NoiseModel(.8,.1,.2)
        values = posterior(Case.from_dict(minimal('pass')), model)
        self.assertAlmostEqual(values['edge'], .8*.2/(.8*.2+.2*.9))
        self.assertLess(values['edge'], 1-model.healthy_prior)

    def test_missing_is_not_a_pass(self):
        case = Case.from_dict(minimal('missing'))
        model = NoiseModel(.8,.1,.2)
        self.assertEqual(posterior(case, model), posterior(replace(case, observations=()),model))
        self.assertEqual(diagnose(case, model)['status'],'insufficient_evidence')

    def test_log_space_stays_finite_with_many_observations(self):
        case = Case.from_dict(minimal())
        probes = tuple(replace(case.observations[0], id=f'p{i}') for i in range(3000))
        result = posterior(replace(case, observations=probes), NoiseModel())
        self.assertTrue(all(math.isfinite(p) for p in result.values()))
        self.assertAlmostEqual(sum(result.values()),1)

    def test_example_exposes_unidentifiable_links(self):
        result = diagnose(Case.from_dict(read_json(ROOT/'examples/ambiguous.json')))
        self.assertEqual(result['status'],'ambiguous')
        self.assertEqual(result['indistinguishable_from_top'],['core','database'])
        self.assertEqual(result['next_probes'][0]['id'],'probe-core')
        self.assertTrue(all(p['supports_top_vs_runner_up_log_bayes_factor']==0
                            for p in result['evidence_comparison']['probes']))

    def test_additional_probe_separates_candidates(self):
        result = diagnose(Case.from_dict(read_json(ROOT/'examples/after-probe.json')))
        self.assertEqual(result['ranking'][0]['hypothesis'],'core')
        self.assertEqual(result['status'],'candidate')
        self.assertGreater(result['ranking'][0]['probability'],.95)
        self.assertEqual(result['indistinguishable_from_top'],['core'])

    def test_counterfactual_topology_changes_answer_with_same_alarms(self):
        raw = read_json(ROOT/'examples/after-probe.json')
        modified = copy.deepcopy(raw)
        for link in modified['links']:
            if link['id']=='core':
                link['source'],link['target']='r2','db'
            elif link['id']=='database':
                link['source'],link['target']='r1','r2'
        self.assertEqual(raw['observations'],modified['observations'])
        self.assertEqual(diagnose(Case.from_dict(modified))['ranking'][0]['hypothesis'],'database')

    def test_node_renaming_and_link_order_do_not_change_probabilities(self):
        raw=read_json(ROOT/'examples/after-probe.json')
        modified=copy.deepcopy(raw)
        for link in modified['links']:
            link['source']='renamed-'+link['source']
            link['target']='renamed-'+link['target']
        modified['links'].reverse()
        for probe in modified['observations']+modified['available_probes']:
            probe['path']=['renamed-'+node for node in probe['path']]
        self.assertEqual(posterior(Case.from_dict(raw),NoiseModel()),posterior(Case.from_dict(modified),NoiseModel()))

    def test_information_gain_equals_enumerated_expected_entropy_reduction(self):
        case = Case.from_dict(minimal('missing'))
        model=NoiseModel(.8,.1,.2)
        probabilities=posterior(case,model)
        probe=case.available_probes[0]
        observed=replace(probe,outcome='fail')
        fail=posterior(replace(case,observations=(observed,)),model)
        passed=posterior(replace(case,observations=(replace(observed,outcome='pass'),)),model)
        pf=.8*.8+.2*.1
        expected=entropy(probabilities.values())-pf*entropy(fail.values())-(1-pf)*entropy(passed.values())
        self.assertAlmostEqual(probe_information(probabilities,probe,model)['expected_gain_bits'],expected)

    def test_probe_cost_changes_recommendation(self):
        case=Case.from_dict(read_json(ROOT/'examples/ambiguous.json'))
        changed=replace(case,available_probes=tuple(replace(p,cost=100) if p.id=='probe-core' else p for p in case.available_probes))
        self.assertEqual(diagnose(changed)['next_probes'][0]['id'],'probe-database')

    def test_equivalent_likelihoods_remain_ambiguous_despite_strong_prior(self):
        raw=minimal()
        raw['links'].append({'id':'unseen','source':'b','target':'c'})
        raw['observations'][0]['outcome']='pass'
        result=diagnose(Case.from_dict(raw),NoiseModel(.9,.03,.999))
        self.assertEqual(result['status'],'ambiguous')
        self.assertEqual(result['indistinguishable_from_top'],[HEALTHY,'unseen'])

    def test_parameter_fit_uses_beta_counts(self):
        rows=[(Case.from_dict(minimal('fail')),'edge')]*8
        rows += [(Case.from_dict(minimal('pass')),'edge')]*2
        rows += [(Case.from_dict(minimal('fail')),HEALTHY)]
        rows += [(Case.from_dict(minimal('pass')),HEALTHY)]*9
        model=fit_noise(rows)
        self.assertAlmostEqual(model.hit_failure_probability,9/12)
        self.assertAlmostEqual(model.background_failure_probability,2/12)
        self.assertAlmostEqual(model.healthy_prior,11/22)

    def test_fit_rejects_unidentifiable_noise_parameters(self):
        with self.assertRaises(ValueError):
            fit_noise([(Case.from_dict(minimal()),HEALTHY)])

    def test_confidence_threshold_is_checked(self):
        case=Case.from_dict(minimal())
        for value in [0,1,float('nan'),float('inf'),True]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                diagnose(case,decision_threshold=value)
