import pointCollection as pc
import numpy as np
import h5py
import re
import glob
import os


def read_DEM_biases(save_file, DEM_roots=None):

    bias_parsers={'jitter':{'re':re.compile(r'sensor_(.*)_jitter_x_atc'),
                           'fmt':'/jitter_bias/sensor_%d_jitter_x_atc'},
                 'stripe':{'re':re.compile(r'sensor_(.*)_stripe_y_atc'),
                       'fmt':'/jitter_bias/sensor_%d_stripe_y_atc'}
                 }

    bias_models={}

    with h5py.File(save_file,'r') as fh:
        bgroup = fh['jitter_bias']
        groups = list(bgroup.keys())
        for group in groups:
            for b_type, bias_parser in bias_parsers.items():
                m = bias_parser['re'].search(group)
                if m is not None:
                    try:
                        sensor = int(m.groups()[0])
                    except:
                        sensor = float(m.groups()[0])
                    if sensor not in bias_models:
                        bias_models[sensor]={}
                    bias_models[sensor][b_type] = {
                        'basis_vectors' : np.array(bgroup[group].attrs['xform_basis_vectors']),
                        'xform_origin'  : np.array(bgroup[group].attrs['xform_origin'])
                    }
                    bias_models[sensor][b_type]['data'] = pc.grid.data().from_h5(save_file, group='jitter_bias/'+group)
                    bias_models[sensor][b_type]['data'].z = np.squeeze(bias_models[sensor][b_type]['data'].z)
                    continue

        DEM_filenames={}
        DEM_sensors = list(bias_models.keys())
        for sensor in DEM_sensors:
            this_DEM_file = str(fh['meta/sensors'].attrs[f'sensor_{int(sensor)}'])
            DEM_filenames[sensor] = this_DEM_file

    if DEM_roots is not None:
        file_for_sensor={}
        for sensor, this_DEM_file in DEM_filenames.items():
            for DEM_root in DEM_roots:
                fname = glob.glob(os.path.join(DEM_root,+this_DEM_file))
                if len(fname) > 0 :
                    file_for_sensor[sensor]=fname[0]
                    break
    SB = pc.data().from_h5(save_file, group='slope_bias')
    B = pc.data().from_h5(save_file, group='bias')

    return B, SB, bias_models, DEM_filenames

def eval_DEM_biases(D, bias_models,  B=None, SB=None):

    D.assign( jitter_bias = 0 )
    D.assign( stripe_bias = 0 )

    for ss in bias_models.keys():
        ii = np.flatnonzero(D.sensor==ss)
        bms = bias_models[ss]
        for bias_type, bm in bms.items():
            if bias_type=='jitter':
                vec_ind=0
            else:
                vec_ind=1
            x0 = bm['xform_origin']
            bv = bm['basis_vectors']
            #print(np.array(bv))
            this_coord = (np.c_[D.x[ii]-x0[0], D.y[ii]-x0[1]] @ np.array(bv))[:, vec_ind]
            #print([np.min(this_coord), np.max(this_coord)])
            #print(this_coord.shape)
            #print('----')
            #this_coord = (D.x[ii] - x0[0])*bv[vec_ind, 0] + (D.y[ii] - x0[1])*bv[vec_ind,1]
            getattr(D, bias_type+'_bias')[ii] = np.interp(this_coord, bm['data'].x, bm['data'].z)
    if B is not None:
        D.assign(bias=0)
        for id, vv in zip(B.ID, B.val):
            D.bias[D.bias_ID==id] = vv

    if SB is not None:
        D.assign(slope_bias=0)
        for sensor, bx, by in zip(SB.sensors, SB.x_slope, SB.y_slope):
            jj = D.sensor==sensor
            x0 = np.nanmean(D.x[jj])
            y0 = np.nanmean(D.y[jj])
            D.slope_bias[jj] = (bx*(D.x[jj]-x0) + by*(D.y[jj]-y0))/1000

def eval_surf(D, save_file=None):

    D.assign(dz = pc.grid.data().from_h5(save_file, group='dz', fields=['dz']).interp(D.x, D.y, D.time, field='dz'))
    D.assign(z0 = pc.grid.data().from_h5(save_file, group='z0', fields=['z0']).interp(D.x, D.y, field='z0'))
    D.assign(z_m = D.z0 + D.dz)


def eval_lag(DD, ii=None, xy0=None, dzL=None, lag_epoch=None):
    DD.assign( dzL = 0 )
    if ii is None:
        ii = range(DD.size)
    if xy0 is not None:
        x0 = xy0.interp(DD.x[ii], DD.y[ii], DD.t[ii] - lag_epoch,  field='x0')
        y0 = xy0.interp(DD.x[ii], DD.y[ii], DD.t[ii] - lag_epoch,  field='y0')
        DD.dzL[ii] = dzL.interp(x0.ravel(),y0.ravel(), field='dz').reshape(DD.x.shape)


# TBD: z0 and dz
