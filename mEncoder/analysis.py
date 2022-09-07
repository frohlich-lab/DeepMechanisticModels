import petab
import numpy as np


def process_simulation_chi2(evaluations, res, conditions, sample, model_type,
                       alpha, hidden_layers):
    splits = {
        'dyn': (conditions[petab.PREEQUILIBRATION_CONDITION_ID] == sample) &
        (conditions[petab.SIMULATION_CONDITION_ID] != sample),
        'stat': (conditions[petab.PREEQUILIBRATION_CONDITION_ID] == sample) &
        (conditions[petab.SIMULATION_CONDITION_ID] == sample),
    }
    for name, split in splits.items():
        ics = np.where(split)[0]
        chi2 = 0
        llh = 0
        for ic in ics:
            chi2 += res['rdatas'][ic].chi2
            llh += res['rdatas'][ic].llh
        evaluations.append({
            'chi2': chi2,
            'llh': llh,
            'sample': f'{sample}_{name}',
            'type': model_type,
            'alpha': alpha,
            'layers': hidden_layers,
        })
