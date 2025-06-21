# This file will be used for RL model validation.

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import json
from collections import defaultdict

# Adjust path to import from src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import RL components
from src.environment.aluminum_melting_env_7 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent


class RLValidator:
    def __init__(self, output_dir="results/rl_validation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize environment
        self.env = AluminumMeltingEnvironment()
        
        # Available models
        self.models_dir = "models"
        self.available_models = [f for f in os.listdir(self.models_dir) 
                               if f.startswith("dqn_final_model") and f.endswith(".pth")]
        
        # Results storage
        self.validation_results = {}
        
    def load_model(self, model_path):
        """Load a trained RL model with automatic architecture detection"""
        try:
            # Load the saved state dict to inspect architecture
            saved_state = torch.load(model_path, map_location='cpu')
            
            # Try to determine architecture from the saved weights
            first_layer_shape = saved_state['0.weight'].shape
            last_layer_shape = saved_state['6.weight'].shape
            
            state_dim = first_layer_shape[1]  # Input dimension
            action_dim = last_layer_shape[0]  # Output dimension
            
            print(f"  Detected architecture: state_dim={state_dim}, action_dim={action_dim}")
            
            # Create agent with detected dimensions
            agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
            agent.model.load_state_dict(saved_state)
            agent.model.eval()
            agent.epsilon = 0.0  # No exploration during validation
            print(f"  ✅ Successfully loaded model from {model_path}")
            return agent
        except Exception as e:
            print(f"  ❌ Error loading model from {model_path}: {e}")
            return None
    
    def evaluate_single_episode(self, agent, episode_id=1, max_steps=1000):
        """Evaluate agent performance for a single episode"""
        state = self.env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        
        # Determine action dimension of the agent
        action_dim = agent.action_dim
        
        # Handle state dimension mismatch
        if len(state) < agent.state_dim:
            # Pad state with zeros if agent expects more dimensions
            padding = np.zeros(agent.state_dim - len(state))
            state = np.concatenate([state, padding])
        elif len(state) > agent.state_dim:
            # Truncate state if agent expects fewer dimensions
            state = state[:agent.state_dim]
        
        # Track detailed metrics
        episode_data = {
            'steps': [],
            'states': [],
            'actions': [],
            'rewards': [],
            'temperatures': [],
            'powers': [],
            'energies': [],
            'action_counts': {i: 0 for i in range(action_dim)}
        }
        
        while not done and steps < max_steps:
            action = agent.select_action(state)
            
            # Map action if agent has different action space than environment
            if action_dim == 2:
                # Map 2-action space to 5-action space
                # 0: increase power, 1: decrease power -> map to env actions
                env_action = 1 if action == 0 else 3  # mild increase or mild decrease
            elif action_dim == 5:
                # Direct mapping for 5-action space
                env_action = action
            else:
                # For other cases, use modulo to map to valid environment actions
                env_action = action % 5
            
            next_state, reward, done = self.env.step(env_action)
            
            # Handle state dimension mismatch for next_state
            if len(next_state) < agent.state_dim:
                # Pad state with zeros if agent expects more dimensions
                padding = np.zeros(agent.state_dim - len(next_state))
                next_state = np.concatenate([next_state, padding])
            elif len(next_state) > agent.state_dim:
                # Truncate state if agent expects fewer dimensions
                next_state = next_state[:agent.state_dim]
            
            # Record data (use original environment state for metrics)
            original_next_state = self.env.state
            episode_data['steps'].append(steps)
            episode_data['states'].append(state.copy())
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['temperatures'].append(original_next_state['temperature'])  # Temperature
            episode_data['powers'].append(original_next_state['power'])             # Power
            episode_data['energies'].append(original_next_state['energy_consumption'])  # Energy
            episode_data['action_counts'][action] += 1
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Calculate episode statistics
        episode_stats = {
            'episode_id': episode_id,
            'total_steps': steps,
            'total_reward': total_reward,
            'avg_reward_per_step': total_reward / steps if steps > 0 else 0,
            'final_temperature': episode_data['temperatures'][-1] if episode_data['temperatures'] else 0,
            'avg_temperature': np.mean(episode_data['temperatures']) if episode_data['temperatures'] else 0,
            'max_temperature': np.max(episode_data['temperatures']) if episode_data['temperatures'] else 0,
            'min_temperature': np.min(episode_data['temperatures']) if episode_data['temperatures'] else 0,
            'avg_power': np.mean(episode_data['powers']) if episode_data['powers'] else 0,
            'final_energy': episode_data['energies'][-1] if episode_data['energies'] else 0,
            'action_distribution': {f'action_{i}_pct': (episode_data['action_counts'][i] / steps * 100) if steps > 0 else 0 
                                   for i in range(action_dim)},
            'episode_data': episode_data
        }
        
        return episode_stats
    
    def evaluate_model(self, model_path, num_episodes=50):
        """Evaluate a model across multiple episodes"""
        print(f"\n🔍 Evaluating model: {model_path}")
        
        # Load model
        agent = self.load_model(model_path)
        if agent is None:
            return None
        
        # Run multiple episodes
        all_episodes = []
        episode_rewards = []
        episode_lengths = []
        episode_energies = []
        
        for ep in range(num_episodes):
            episode_stats = self.evaluate_single_episode(agent, ep + 1)
            all_episodes.append(episode_stats)
            episode_rewards.append(episode_stats['total_reward'])
            episode_lengths.append(episode_stats['total_steps'])
            episode_energies.append(episode_stats['final_energy'])
            
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep + 1}/{num_episodes} completed")
        
        # Calculate aggregate statistics
        model_results = {
            'model_name': os.path.basename(model_path),
            'num_episodes': num_episodes,
            'episodes': all_episodes,
            'aggregate_stats': {
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'avg_episode_length': np.mean(episode_lengths),
                'std_episode_length': np.std(episode_lengths),
                'avg_final_energy': np.mean(episode_energies),
                'std_final_energy': np.std(episode_energies),
                'success_rate': len([r for r in episode_rewards if r > 0]) / len(episode_rewards) * 100
            }
        }
        
        print(f"  ✅ Model evaluation completed:")
        print(f"    Average Reward: {model_results['aggregate_stats']['avg_reward']:.2f} ± {model_results['aggregate_stats']['std_reward']:.2f}")
        print(f"    Average Episode Length: {model_results['aggregate_stats']['avg_episode_length']:.1f} steps")
        print(f"    Success Rate: {model_results['aggregate_stats']['success_rate']:.1f}%")
        
        return model_results
    
    def compare_models(self, num_episodes=50):
        """Compare all available models"""
        print(f"\n🚀 Starting RL Model Validation")
        print(f"Found {len(self.available_models)} models to validate")
        print(f"Results will be saved to: {self.output_dir}")
        
        comparison_results = {}
        
        for model_file in sorted(self.available_models):
            model_path = os.path.join(self.models_dir, model_file)
            model_results = self.evaluate_model(model_path, num_episodes)
            
            if model_results:
                comparison_results[model_file] = model_results
        
        self.validation_results = comparison_results
        return comparison_results
    
    def create_visualizations(self):
        """Create visualization plots for validation results"""
        if not self.validation_results:
            print("❌ No validation results to visualize")
            return
        
        print("\n📊 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model Comparison - Average Rewards
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL Model Validation Results', fontsize=16, fontweight='bold')
        
        model_names = []
        avg_rewards = []
        std_rewards = []
        avg_energies = []
        avg_lengths = []
        
        for model_name, results in self.validation_results.items():
            model_names.append(model_name.replace('dqn_final_model_', 'Model ').replace('.pth', ''))
            avg_rewards.append(results['aggregate_stats']['avg_reward'])
            std_rewards.append(results['aggregate_stats']['std_reward'])
            avg_energies.append(results['aggregate_stats']['avg_final_energy'])
            avg_lengths.append(results['aggregate_stats']['avg_episode_length'])
        
        # Average Rewards with error bars
        axes[0, 0].bar(model_names, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Average Reward per Model')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average Final Energy
        axes[0, 1].bar(model_names, avg_energies, alpha=0.7, color='orange')
        axes[0, 1].set_title('Average Final Energy Consumption')
        axes[0, 1].set_ylabel('Energy (kWh)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average Episode Length
        axes[1, 0].bar(model_names, avg_lengths, alpha=0.7, color='green')
        axes[1, 0].set_title('Average Episode Length')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward Distribution (Box Plot)
        all_rewards_data = []
        labels = []
        for model_name, results in self.validation_results.items():
            rewards = [ep['total_reward'] for ep in results['episodes']]
            all_rewards_data.append(rewards)
            labels.append(model_name.replace('dqn_final_model_', 'M').replace('.pth', ''))
        
        axes[1, 1].boxplot(all_rewards_data, labels=labels)
        axes[1, 1].set_title('Reward Distribution by Model')
        axes[1, 1].set_ylabel('Total Reward')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Best Model Detailed Analysis
        best_model_name = max(self.validation_results.keys(), 
                             key=lambda k: self.validation_results[k]['aggregate_stats']['avg_reward'])
        best_model_results = self.validation_results[best_model_name]
        
        self.create_detailed_analysis(best_model_name, best_model_results)
        
        print(f"  ✅ Visualizations saved to {self.output_dir}")
    
    def create_detailed_analysis(self, model_name, results):
        """Create detailed analysis for the best performing model"""
        print(f"  📈 Creating detailed analysis for {model_name}")
        
        # Select a representative episode (median performance)
        episode_rewards = [ep['total_reward'] for ep in results['episodes']]
        median_idx = np.argsort(episode_rewards)[len(episode_rewards) // 2]
        representative_episode = results['episodes'][median_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Detailed Analysis: {model_name} (Representative Episode)', fontsize=16, fontweight='bold')
        
        episode_data = representative_episode['episode_data']
        
        # Temperature over time
        axes[0, 0].plot(episode_data['steps'], episode_data['temperatures'], 'b-', linewidth=2)
        axes[0, 0].set_title('Temperature Profile')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power over time
        axes[0, 1].plot(episode_data['steps'], episode_data['powers'], 'r-', linewidth=2)
        axes[0, 1].set_title('Power Profile')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Power (kW)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy consumption over time
        axes[1, 0].plot(episode_data['steps'], episode_data['energies'], 'g-', linewidth=2)
        axes[1, 0].set_title('Cumulative Energy Consumption')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Energy (kWh)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Action distribution
        action_counts = representative_episode['episode_data']['action_counts']
        
        # Get action labels based on action space size
        if len(action_counts) == 2:
            actions = ['Increase Power', 'Decrease Power']
            colors = ['skyblue', 'lightcoral']
        elif len(action_counts) == 5:
            actions = ['Inc Strong', 'Inc Mild', 'Maintain', 'Dec Mild', 'Dec Strong']
            colors = ['red', 'orange', 'yellow', 'lightblue', 'blue']
        else:
            actions = [f'Action {i}' for i in range(len(action_counts))]
            colors = plt.cm.Set3(np.linspace(0, 1, len(action_counts)))
        
        counts = [action_counts[i] for i in range(len(action_counts))]
        
        axes[1, 1].pie(counts, labels=actions, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Action Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'detailed_analysis_{model_name.replace(".pth", "")}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save validation results to files"""
        print("\n💾 Saving validation results...")
        
        # Save detailed results as JSON
        results_file = os.path.join(self.output_dir, 'validation_results.json')
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = self.make_serializable(self.validation_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary statistics as CSV
        summary_data = []
        for model_name, results in self.validation_results.items():
            summary_data.append({
                'Model': model_name,
                'Avg_Reward': results['aggregate_stats']['avg_reward'],
                'Std_Reward': results['aggregate_stats']['std_reward'],
                'Min_Reward': results['aggregate_stats']['min_reward'],
                'Max_Reward': results['aggregate_stats']['max_reward'],
                'Avg_Episode_Length': results['aggregate_stats']['avg_episode_length'],
                'Avg_Final_Energy': results['aggregate_stats']['avg_final_energy'],
                'Success_Rate': results['aggregate_stats']['success_rate']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.output_dir, 'validation_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        
        # Create validation report
        self.create_validation_report()
        
        print(f"  ✅ Results saved to:")
        print(f"    - {results_file}")
        print(f"    - {summary_file}")
        print(f"    - {os.path.join(self.output_dir, 'validation_report.md')}")
    
    def make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    def create_validation_report(self):
        """Create a markdown validation report"""
        report_file = os.path.join(self.output_dir, 'validation_report.md')
        
        # Find best performing model
        best_model_name = max(self.validation_results.keys(), 
                             key=lambda k: self.validation_results[k]['aggregate_stats']['avg_reward'])
        best_model = self.validation_results[best_model_name]
        
        report_content = f"""# RL Model Validation Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report presents the validation results for {len(self.validation_results)} RL models trained for the aluminum melting process optimization.

## Models Evaluated

{chr(10).join([f"- {model}" for model in self.validation_results.keys()])}

## Best Performing Model

**Model:** {best_model_name}

### Performance Metrics:
- **Average Reward:** {best_model['aggregate_stats']['avg_reward']:.2f} ± {best_model['aggregate_stats']['std_reward']:.2f}
- **Success Rate:** {best_model['aggregate_stats']['success_rate']:.1f}%
- **Average Episode Length:** {best_model['aggregate_stats']['avg_episode_length']:.1f} steps
- **Average Final Energy:** {best_model['aggregate_stats']['avg_final_energy']:.2f} kWh

## Model Comparison

| Model | Avg Reward | Std Reward | Success Rate (%) | Avg Energy (kWh) |
|-------|------------|------------|------------------|------------------|
"""
        
        for model_name, results in sorted(self.validation_results.items()):
            stats = results['aggregate_stats']
            report_content += f"| {model_name} | {stats['avg_reward']:.2f} | {stats['std_reward']:.2f} | {stats['success_rate']:.1f} | {stats['avg_final_energy']:.2f} |\n"
        
        report_content += f"""
## Key Findings

1. **Best Model:** {best_model_name} achieved the highest average reward of {best_model['aggregate_stats']['avg_reward']:.2f}
2. **Consistency:** The model with the lowest reward variance was determined based on standard deviation
3. **Energy Efficiency:** Energy consumption patterns varied across models, indicating different control strategies

## Visualizations

The following visualizations are available in this directory:
- `model_comparison.png`: Comparison of all models across key metrics
- `detailed_analysis_{best_model_name.replace('.pth', '')}.png`: Detailed analysis of the best performing model

## Validation Methodology

- **Episodes per Model:** {best_model['num_episodes']}
- **Environment:** AluminumMeltingEnvironment
- **Evaluation Mode:** Epsilon = 0.0 (no exploration)
- **Metrics:** Total reward, episode length, energy consumption, action distribution

## Recommendations

1. **Production Deployment:** Consider {best_model_name} for production deployment based on validation results
2. **Further Testing:** Conduct additional validation with different scenarios and longer episodes
3. **Monitoring:** Implement continuous monitoring of model performance in production environment
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)


def main():
    """Main validation execution"""
    print("🔬 RL Model Validation System")
    print("=" * 50)
    
    # Initialize validator
    validator = RLValidator()
    
    # Run validation
    results = validator.compare_models(num_episodes=50)
    
    if results:
        # Create visualizations
        validator.create_visualizations()
        
        # Save results
        validator.save_results()
        
        print("\n🎉 Validation Complete!")
        print(f"📁 Results saved to: {validator.output_dir}")
        
        # Display summary
        print("\n📊 Validation Summary:")
        for model_name, model_results in results.items():
            stats = model_results['aggregate_stats']
            print(f"  {model_name}: Avg Reward = {stats['avg_reward']:.2f}, Success Rate = {stats['success_rate']:.1f}%")
    else:
        print("❌ No models were successfully validated")


if __name__ == "__main__":
    main()
