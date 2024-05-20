import numpy as np
from PIL import Image
from source.oscillators import PeripheralOscillator
from source.onn_model import OnnModel2D, OnnSelectiveAttentionModule2D
from source.utils import display_image


def test_get_synchronization_state(onn_sel_att_module: OnnSelectiveAttentionModule2D):
    onn_sel_att_module.get_synchonization_state()
    sync_states = np.asarray(onn_sel_att_module.synchronization_states)
    assert np.isin(sync_states, [0, 1]).all() == True

def test_check_synchronization_state_1(onn_sel_att_module: OnnSelectiveAttentionModule2D):
    for ensemble in onn_sel_att_module.periferal_oscillators:
        for po in ensemble:
            po.phase = onn_sel_att_module.central_oscillator.phase

    onn_sel_att_module.periferal_oscillators[0][0].phase = -1000
    state, state_id = onn_sel_att_module.check_synchronization_state()
    assert state == "ps"
    assert state_id != 0

def test_check_synchronization_state_2(onn_sel_att_module: OnnSelectiveAttentionModule2D):
    for ensemble in onn_sel_att_module.periferal_oscillators:
        for po in ensemble:
            po.phase = onn_sel_att_module.central_oscillator.phase

    state, state_id = onn_sel_att_module.check_synchronization_state()
    assert state == "gs"
    assert state_id != 0

def test_sel_att_module_run():

    image_path = r'tests\test_images\simple_scene.jpg' 
    img = Image.open(image_path)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    img_array = img_array.reshape((3, 1024, 1024))

    onn_sel_att_module = OnnSelectiveAttentionModule2D("SelAtt")
    
    selected_area = onn_sel_att_module.run(img=img_array)

    print(selected_area)

    print(img_array)
    
    img_array[0][img_array[0] != selected_area[0]] = 0
    img_array[1][img_array[1] != selected_area[1]] = 0
    img_array[2][img_array[2] != selected_area[2]] = 0

    # img_band = img_band[~np.all(img_band == 0, axis=1)]
    # img_band = img_band[:,~(img_band==0).all(0)]

    display_image(img_array.reshape((img_array.shape[1], img_array.shape[2], img_array.shape[0])), [0, 1, 2])

    display_image(selected_area.reshape((selected_area.shape[1], selected_area.shape[2], selected_area.shape[0])), [0, 1, 2])
