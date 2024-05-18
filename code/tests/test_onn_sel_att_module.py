import numpy as np
from source.oscillators import PeripheralOscillator
from source.onn_model import OnnModel2D, OnnSelectiveAttentionModule2D


def test_get_synchronization_state(onn_sel_att_module: OnnSelectiveAttentionModule2D):
    onn_sel_att_module.get_synchonization_state()
    sync_states = np.asarray(onn_sel_att_module.synchronization_states)
    assert np.isin(sync_states, [0, 1]).all() == True

def test_check_synchronization_state_1(onn_sel_att_module: OnnSelectiveAttentionModule2D):
    for ensemble in onn_sel_att_module.periferal_oscillators:
        for po in ensemble:
            po.phase = onn_sel_att_module.central_oscillator.phase

    onn_sel_att_module.periferal_oscillators[0][0].phase = -1
    state, state_id = onn_sel_att_module.check_synchronization_state()
    assert state == "ps"
    assert state_id != 0

def test_check_synchronization_state_2(onn_sel_att_module: OnnSelectiveAttentionModule2D):
    print(onn_sel_att_module.periferal_oscillators[0][0].phase)
    for ensemble in onn_sel_att_module.periferal_oscillators:
        for po in ensemble:
            po.phase = onn_sel_att_module.central_oscillator.phase

    state, state_id = onn_sel_att_module.check_synchronization_state()
    assert state == "gs"
    assert state_id != 0
    