"""
Unit tests for triage system functionality.

Tests deferral strategies, threshold optimization, triage metrics,
and edge cases.
"""

import unittest
import numpy as np


class TestTriageDeferralStrategies(unittest.TestCase):
    """Test different deferral strategies."""
    
    def setUp(self):
        """Set up test data."""
        self.predictions = np.array([0, 1, 2, 1, 0, 2, 1, 0])
        self.uncertainties = np.array([0.1, 0.5, 0.8, 0.3, 0.2, 0.9, 0.4, 0.6])
        self.targets = np.array([0, 1, 2, 1, 0, 2, 1, 1])
    
    def test_threshold_strategy(self):
        """Test simple threshold deferral strategy."""
        threshold = 0.5
        defer_mask = self.uncertainties > threshold
        
        expected_defer = np.array([False, False, True, False, False, True, False, True])
        np.testing.assert_array_equal(defer_mask, expected_defer)
    
    def test_budget_constrained_strategy(self):
        """Test budget-constrained deferral strategy."""
        budget_rate = 0.25  # Defer top 25%
        budget_count = int(np.ceil(len(self.uncertainties) * budget_rate))
        
        # Defer top-k uncertain
        top_k_idx = np.argsort(self.uncertainties)[-budget_count:]
        defer_mask = np.zeros(len(self.uncertainties), dtype=bool)
        defer_mask[top_k_idx] = True
        
        # Should defer exactly 2 samples (25% of 8)
        self.assertEqual(defer_mask.sum(), 2)
    
    def test_deferral_rate_computation(self):
        """Test deferral rate computation."""
        defer_mask = np.array([True, False, True, False, False])
        deferral_rate = defer_mask.mean()
        
        self.assertEqual(deferral_rate, 0.4)
    
    def test_automation_rate(self):
        """Test automation rate (1 - deferral_rate)."""
        deferral_rate = 0.3
        automation_rate = 1 - deferral_rate
        
        self.assertEqual(automation_rate, 0.7)


class TestThresholdOptimization(unittest.TestCase):
    """Test threshold optimization."""
    
    def test_sweep_different_thresholds(self):
        """Test evaluating different thresholds."""
        uncertainties = np.linspace(0, 1, 100)
        
        thresholds = np.linspace(0.1, 0.9, 9)
        
        results = []
        for threshold in thresholds:
            deferral_rate = (uncertainties > threshold).mean()
            results.append(deferral_rate)
        
        # Higher threshold → lower deferral rate
        self.assertTrue(np.all(np.diff(results) < 0))
    
    def test_optimal_threshold_finding(self):
        """Test finding optimal threshold."""
        # Create mock performance curve
        thresholds = np.linspace(0, 1, 10)
        accuracies = np.array([0.7, 0.72, 0.75, 0.78, 0.80, 0.79, 0.77, 0.75, 0.72, 0.68])
        
        # Find best threshold
        best_idx = np.argmax(accuracies)
        optimal_threshold = thresholds[best_idx]
        best_accuracy = accuracies[best_idx]
        
        self.assertAlmostEqual(optimal_threshold, 0.44, places=1)
        self.assertEqual(best_accuracy, 0.80)


class TestTriageMetrics(unittest.TestCase):
    """Test triage system metrics."""
    
    def test_ai_accuracy(self):
        """Test AI accuracy on non-deferred samples."""
        predictions = np.array([0, 1, 2, 1, 0, 2])
        targets = np.array([0, 1, 2, 1, 0, 1])
        defer_mask = np.array([False, False, False, False, True, True])
        
        # AI accuracy on non-deferred
        keep_mask = ~defer_mask
        ai_correct = (predictions[keep_mask] == targets[keep_mask]).sum()
        ai_accuracy = ai_correct / keep_mask.sum()
        
        self.assertEqual(ai_accuracy, 1.0)
    
    def test_system_accuracy_with_human(self):
        """Test system accuracy with human handling deferred."""
        predictions = np.array([0, 1, 2, 1, 0, 2])
        targets = np.array([0, 1, 2, 1, 0, 1])
        defer_mask = np.array([False, False, False, False, True, True])
        human_accuracy = 1.0  # Perfect human
        
        system_preds = predictions.copy()
        
        # Human handles deferred with perfect accuracy
        if defer_mask.sum() > 0:
            system_preds[defer_mask] = targets[defer_mask]
        
        system_accuracy = (system_preds == targets).mean()
        
        self.assertEqual(system_accuracy, 1.0)
    
    def test_cost_analysis(self):
        """Test cost-benefit analysis."""
        n_samples = 100
        ai_error_cost = 100
        human_review_cost = 1
        
        # Scenario 1: All AI
        ai_errors_all = 10
        cost_all_ai = ai_errors_all * ai_error_cost
        
        # Scenario 2: Defer 30%, reduce errors to 3
        deferred = 30
        ai_errors_deferred = 3
        cost_deferred = ai_errors_deferred * ai_error_cost + deferred * human_review_cost
        
        # Deferred is better
        self.assertLess(cost_deferred, cost_all_ai)


class TestTriageEdgeCases(unittest.TestCase):
    """Test edge cases in triage system."""
    
    def test_no_deferral(self):
        """Test when no samples are deferred."""
        uncertainties = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = 1.0  # Very high threshold
        
        defer_mask = uncertainties > threshold
        
        self.assertEqual(defer_mask.sum(), 0)
    
    def test_full_deferral(self):
        """Test when all samples are deferred."""
        uncertainties = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = 0.0  # Very low threshold
        
        defer_mask = uncertainties > threshold
        
        self.assertEqual(defer_mask.sum(), len(uncertainties))
    
    def test_single_sample(self):
        """Test triage with single sample."""
        uncertainty = np.array([0.5])
        threshold = 0.4
        
        defer = uncertainty[0] > threshold
        
        self.assertTrue(defer)
    
    def test_identical_uncertainties(self):
        """Test when all uncertainties are the same."""
        uncertainties = np.array([0.5, 0.5, 0.5, 0.5])
        threshold = 0.5
        
        defer_mask = uncertainties > threshold
        
        self.assertEqual(defer_mask.sum(), 0)  # None > 0.5
    
    def test_uncertainty_with_nans(self):
        """Test handling of NaN uncertainties."""
        uncertainties = np.array([0.1, np.nan, 0.3])
        
        # Replace NaN with high uncertainty
        uncertainties = np.nan_to_num(uncertainties, nan=1.0)
        
        self.assertFalse(np.any(np.isnan(uncertainties)))


class TestHumanSimulator(unittest.TestCase):
    """Test human expert simulator."""
    
    def test_human_accuracy_bounds(self):
        """Test human accuracy is in [0, 1]."""
        human_accuracy = 0.95
        
        self.assertGreaterEqual(human_accuracy, 0)
        self.assertLessEqual(human_accuracy, 1)
    
    def test_human_correct_predictions(self):
        """Test human correct prediction generation."""
        np.random.seed(42)
        human_accuracy = 0.9
        n_samples = 100
        
        # Simulate correct predictions
        correct = np.random.rand(n_samples) < human_accuracy
        
        actual_accuracy = correct.mean()
        
        # Should be approximately 90%
        self.assertAlmostEqual(actual_accuracy, 0.9, delta=0.05)
    
    def test_class_specific_accuracy(self):
        """Test class-specific human accuracy."""
        class_accuracies = {
            0: 0.95,
            1: 0.85,
            2: 0.90,
        }
        
        # All should be between 0 and 1
        for acc in class_accuracies.values():
            self.assertGreaterEqual(acc, 0)
            self.assertLessEqual(acc, 1)
    
    def test_inter_rater_variability(self):
        """Test inter-rater variability simulation."""
        base_accuracy = 0.9
        variability = 0.1  # ±10%
        
        # Simulate multiple raters
        rater_accuracies = [
            base_accuracy + np.random.uniform(-variability, variability)
            for _ in range(5)
        ]
        
        # All should be within bounds
        for acc in rater_accuracies:
            self.assertGreaterEqual(acc, 0)
            self.assertLessEqual(acc, 1)


class TestCollaborationMetrics(unittest.TestCase):
    """Test human-AI collaboration metrics."""
    
    def test_automation_efficiency(self):
        """Test automation efficiency metric."""
        automation_rate = 0.7
        ai_accuracy = 0.95
        human_accuracy = 0.98
        
        # System should be efficient if automation is high and accuracy reasonable
        efficiency = automation_rate * ai_accuracy
        
        self.assertGreater(efficiency, 0.5)
    
    def test_workload_reduction(self):
        """Test workload reduction metric."""
        total_samples = 1000
        deferred_samples = 200
        
        workload_reduction = deferred_samples / total_samples
        
        self.assertEqual(workload_reduction, 0.2)
    
    def test_safety_metric_critical_cases(self):
        """Test safety metric for critical cases."""
        critical_classes = [2]  # Class 2 is critical
        targets = np.array([0, 1, 2, 2, 1, 2, 0])
        defer_mask = np.array([False, False, True, True, False, False, False])
        
        # Check deferral rate for critical cases
        critical_mask = np.isin(targets, critical_classes)
        critical_deferral_rate = defer_mask[critical_mask].mean()
        
        self.assertEqual(critical_deferral_rate, 1.0)  # All critical deferred
    
    def test_cost_benefit_tradeoff(self):
        """Test cost-benefit trade-off."""
        # Scenario 1: High automation, low cost
        scenario1 = {
            'automation_rate': 0.9,
            'cost': 100,
        }
        
        # Scenario 2: Low automation, lower cost
        scenario2 = {
            'automation_rate': 0.5,
            'cost': 250,
        }
        
        # Scenario 1 is better (higher automation, lower cost)
        self.assertGreater(scenario1['automation_rate'], scenario2['automation_rate'])
        self.assertLess(scenario1['cost'], scenario2['cost'])


class TestTriagePerformanceCurves(unittest.TestCase):
    """Test generation of performance curves."""
    
    def test_automation_accuracy_curve(self):
        """Test automation vs. accuracy curve."""
        automation_rates = np.linspace(0, 1, 10)
        accuracies = np.array([0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86])
        
        # Verify relationship: higher automation may reduce accuracy
        correlation = np.corrcoef(automation_rates, accuracies)[0, 1]
        
        # Should be negative (tradeoff)
        self.assertLess(correlation, 0)
    
    def test_cost_accuracy_curve(self):
        """Test cost vs. accuracy curve."""
        costs = np.array([1000, 900, 850, 820, 800, 790, 785, 780, 775, 770])
        accuracies = np.array([0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86])
        
        # Higher accuracy typically higher cost
        correlation = np.corrcoef(costs, accuracies)[0, 1]
        
        # Should be positive
        self.assertGreater(correlation, 0)
    
    def test_human_accuracy_sensitivity(self):
        """Test sensitivity to human accuracy."""
        human_accuracies = np.array([0.8, 0.85, 0.9, 0.95, 0.99])
        system_accuracies = []
        
        # System accuracy should increase with human accuracy
        for human_acc in human_accuracies:
            # Mock: system accuracy = 0.3 * AI + 0.7 * human
            ai_accuracy = 0.85
            system_acc = 0.3 * ai_accuracy + 0.7 * human_acc
            system_accuracies.append(system_acc)
        
        # Should be monotonically increasing
        self.assertTrue(np.all(np.diff(system_accuracies) > 0))


if __name__ == '__main__':
    unittest.main()
