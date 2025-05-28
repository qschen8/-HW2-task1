import itertools
import json
import os
import time
import copy
from train import main as train_model
from train import get_config

EPOCH = 10

search_space = {
    'lr_backbone': [1e-5, 1e-4],
    'lr_fc': [1e-4, 3e-4],
    'batch_size': [64, 128, 256],
    'scheduler_gamma': [0.1, 0.01],
}


keys = search_space.keys()
values = (search_space[key] for key in keys)
config_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

results = []
best_accuracy = 0.0
best_config = {}

exp_dir = f"exp/exp_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(exp_dir, exist_ok=True)

for idx, params in enumerate(config_combinations):
    print(f"\n=== Running trial {idx+1}/{len(config_combinations)} ===")
    print("Hyperparameters:", json.dumps(params, indent=2))
    
    config = get_config()
    current_config = copy.deepcopy(config)
    current_config.update(params)
    
    run_id = f"run_{idx}_{time.strftime('%H%M%S')}"
    current_config['exp_dir'] = os.path.join(exp_dir, run_id)
    
    current_config.update({
        'log_dir': os.path.join(current_config['exp_dir'], 'logs'),
        'checkpoint_dir': os.path.join(current_config['exp_dir'], 'checkpoints'),
        'num_epochs': EPOCH  
    })

    os.makedirs(current_config['log_dir'], exist_ok=True)
    os.makedirs(current_config['checkpoint_dir'], exist_ok=True)

    try:
        best_acc = train_model(current_config)
        
        results.append({
            'config': current_config,
            'best_accuracy': best_acc
        })
        
        with open(os.path.join(current_config['exp_dir'], 'results.json'), 'w') as f:
            json.dump({
                'config': current_config,
                'best_accuracy': best_acc
            }, f, indent=2)

        if best_acc > best_accuracy:
            best_accuracy = best_acc
            best_config = current_config
            print(f"New best accuracy: {best_accuracy:.2%}")

    except Exception as e:
        print(f"Trial {idx+1} failed with error: {str(e)}")
        continue

report = {
    'total_trials': len(config_combinations),
    'completed_trials': len(results),
    'best_accuracy': best_accuracy,
    'best_config': best_config,
    'all_results': results
}

with open(os.path.join(exp_dir, 'final_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

print("\n=== Hyperparameter Search Completed ===")
print(f"Best Accuracy: {best_accuracy:.2%}")
print("Best Configuration:")
print(json.dumps(best_config, indent=2))