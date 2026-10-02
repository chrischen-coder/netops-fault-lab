from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from netops_fault_lab import Case, fit_noise
from netops_fault_lab.benchmark import active_trial, candidate_features, rank, run_benchmark
from netops_fault_lab.synthetic import incident, topologies


class ExperimentTests(unittest.TestCase):
    def test_generator_is_reproducible_and_groups_are_distinct(self):
        graphs=topologies(8,42)
        self.assertEqual(graphs,topologies(8,42))
        self.assertEqual(len({g['topology_id'] for g in graphs}),8)
        self.assertEqual(incident(graphs[0],42,'a'),incident(graphs[0],42,'a'))

    def test_copied_alarms_add_no_new_paths_or_outcomes(self):
        graph=topologies(1,42)[0]
        nominal=incident(graph,101,'same','nominal')
        copied=incident(graph,101,'same','copied_alarms_x4')
        self.assertEqual(nominal['truth'],copied['truth'])
        self.assertEqual(len(copied['case']['observations']),4*len(nominal['case']['observations']))
        for left,right in zip(nominal['case']['observations'],copied['case']['observations'][::4]):
            self.assertEqual(left,right)

    def test_learning_features_do_not_use_case_or_link_identifier_text(self):
        graph=topologies(1,42)[0]
        row=incident(graph,101,'old')
        case=Case.from_dict(row['case'])
        for hypothesis in case.hypotheses:
            self.assertEqual(candidate_features(case,hypothesis),candidate_features(replace(case,id='new',topology_id='new'),hypothesis))

    def test_complete_small_benchmark_has_disjoint_splits_and_fair_active_start(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_benchmark(Path(folder)/'run',seeds=[7],train_topologies=3,test_topologies=2,cases_per_topology=4)
            manifest=result['manifest'][0]
            self.assertFalse(set(manifest['train_topology_ids']) & set(manifest['test_topology_ids']))
            self.assertEqual(result['summary']['nominal']['bayes']['cases'],8)
            self.assertEqual(result['active_probes']['random']['0'],result['active_probes']['information_gain']['0'])
            self.assertTrue((Path(folder)/'run'/'predictions.csv').is_file())

    def test_benchmark_refuses_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                run_benchmark(Path(folder),seeds=[7],train_topologies=3,test_topologies=2,cases_per_topology=4)
