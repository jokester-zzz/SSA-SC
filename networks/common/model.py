from networks.models.LMSCNet import LMSCNet
from networks.models.LMSCNet_SS import LMSCNet_SS
from networks.models.SSCNet_full import SSCNet_full
from networks.models.SSCNet import SSCNet
from networks.models.LMSCNet_SSP import LMSCNet_SSP
from networks.models.BEV_UNet import BEV_UNet
from networks.models.SSA_SC import SSA_SC


def get_model(_cfg, dataset):

    nbr_classes = dataset.nbr_classes
    grid_dimensions = dataset.grid_dimensions
    class_frequencies = dataset.class_frequencies

    selected_model = _cfg._dict['MODEL']['TYPE']

    # LMSCNet ----------------------------------------------------------------------------------------------------------
    if selected_model == 'LMSCNet':
        model = LMSCNet(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
    # ------------------------------------------------------------------------------------------------------------------

    # LMSCNet_SS -------------------------------------------------------------------------------------------------------
    elif selected_model == 'LMSCNet_SS':
        model = LMSCNet_SS(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
    # ------------------------------------------------------------------------------------------------------------------

    # SSCNet_full ------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSCNet_full':
        model = SSCNet_full(class_num=nbr_classes)
    # ------------------------------------------------------------------------------------------------------------------

    # SSCNet -----------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSCNet':
        model = SSCNet(class_num=nbr_classes)
    # ------------------------------------------------------------------------------------------------------------------
    
    # LMSCNet_SSP ------------------------------------------------------------------------------------------------------
    elif selected_model == 'LMSCNet_SSP':
        model = LMSCNet_SSP(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies, pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512, fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------
    
    # BEV_Unet ---------------------------------------------------------------------------------------------------------
    elif selected_model == 'BEV_Unet':
        model = BEV_UNet(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies, pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512, fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------

    # SSA_SC --------------------------------------------------------------------------------------------------------
    elif selected_model == 'SSA_SC':
        model = SSA_SC(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies, pt_model='pointnet', fea_dim=7, pt_pooling='max', kernal_size=3, out_pt_fea_dim=512, fea_compre=32)
    # ------------------------------------------------------------------------------------------------------------------

   

    else:
        assert False, 'Wrong model selected'

    return model



        # model = ptBEVnet(BEV_model, pt_model='pointnet', grid_size=grid_dimensions, fea_dim=7, out_pt_fea_dim=512, kernal_size=3, fea_compre=32)