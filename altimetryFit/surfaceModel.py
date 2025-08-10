#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 11:54:08 2025

@author: ben
"""

import pointCollection as pc
import numpy as np
import os
import h5py
import glob

class surfaceModel(pc.grid.data):
    def __init__(self,
                 x=None, y=None, t=None,
                 reference_time=None,
                 SMB_file=None,
                 velocity_file=None,
                 floating_mask_file=None,
                 floating_mask_value=1,
                 shelf=False,
                 spacing=None,
                 velocity_type='xy01_grids'):
        super().__init__()
        self._xy0_interpolator=None
        self._z0=None
        self._dz=None
        self._lagrangian_dz=None
        self.shelf=shelf
        self.SMB_file=SMB_file
        self.components=['z0','dz']
        self.reference_time=reference_time
        self.x=x
        self.y=y
        self.t=t
        self.shape=None
        self.floating_mask_file=floating_mask_file
        self.floating_mask_value=floating_mask_value
        self.velocity_files=velocity_file
        self.velocity_type=velocity_type
        self.spacing=spacing

    def _read_vel_interpolators(self, velocity_files=None, velocity_type=None):
        if velocity_files is None:
            velocity_files=self.velocity_files
        if velocity_type is None:
            velocity_type=self.velocity_type
        if 'xy01_grids' in velocity_files[0] or velocity_type=='xy01_grids':
            interpolator_save_files=velocity_files
            if not os.path.isfile(interpolator_save_files[0]):
                interpolator_save_files=glob.glob(interpolator_save_files[0])
                if len(interpolator_save_files)==0:
                    raise(FileNotFoundError(f"Interpolator save files {interpolator_save_files} not found"))
        else:
            interpolator_save_files=[os.path.splitext(velocity_files[0])[0] +'_xy01_grids.h5']
        if os.path.isfile(interpolator_save_files[0]):
            self._xy0_interpolator=pc.grid.mosaic().from_list(interpolator_save_files, bounds=self.bounds(), group='xy0')
            with h5py.File(interpolator_save_files[0],'r') as h5f:
                try:
                    self._xy0_interpolator.t += np.array( h5f['xy0'].attrs['lagrangian_epoch'])
                except:
                    pass

    def _match_extent(self, filename):
        if filename.endswith('tif') or filename.endswith('tiff'):
            temp=pc.grid.data().from_geotif(filename, meta_only=True)
        else:
            for group in ['z0','dz']:
                try:
                    temp=pc.grid.data().from_file(filename, group=group, meta_only=True)
                    break
                except:
                    pass
        bds=temp.bounds()
        self.x, self.y  = [np.arange(bb[0], bb[-1]+0.1*self.spacing, self.spacing)
                           for bb in bds]
    def _get_shifted_xy(self, assign=False):
        if np.isscalar(self.t):
            t=[self.t]
        if self._xy0_interpolator is None:
            self._read_vel_interpolators()
        x0 = self._xy0_interpolator.interp(self.x, self.y, t, gridded=True, field='x0')
        y0 = self._xy0_interpolator.interp(self.x, self.y, t, gridded=True, field='y0')
        if assign:
            self.assign(x0=x0, y0=y0)
        else:
            return x0, y0

    def from_tile(self, filename, bounds=None, match_extent=False):

        if match_extent:
            self._match_extent(filename)

        if bounds is None:
            bounds=self.bounds()

        for group in ['z0','dz','lagrangian_dz']:
            try:
                setattr(self, '_'+group, pc.grid.data().from_h5(filename, group=group, bounds=bounds))
            except Exception:
                pass
        return self

    def from_h5_directory(self, directory, bounds=None, match_extent=False, VERBOSE=False):
        """
        Read a model from a directory

        Parameters
        ----------
        directory : str
            Directory in which to look for files.  Should contain z0.h5, dz.h5, and possibly lagrangian_dz.h5 or lag_dz.h5
        bounds : iterable, optional
            The bounds of the returned object: [xmin, xmax], [ymin, ymax]. The default is None.
        match_extent : Boolean, optional
            If true, match the returned object dimension to the files in the directory. The default is False.

        """
        if match_extent:
            self._match_extent(os.path.join(directory, 'z0.h5'))

        if bounds is None:
            bounds=self.bounds()

        for group in ['z0','dz','lagrangian_dz']:
            try:
                this_file = os.path.join(directory, group+'.h5')
                if not os.path.isfile(this_file):
                    if group=='lagrangian_dz':
                        this_file=os.path.join(directory, 'lag_dz.h5')
                        if not os.path.isfile(this_file):
                            continue
                    else:
                        continue
                temp = pc.grid.data().from_h5(this_file, group=group, bounds=bounds)
                if 'cell_area' in temp.fields:
                    mask=(temp.cell_area > 0.001*(temp.x[1]-temp.x[0])*(temp.y[1]-temp.y[0])).astype(float)
                    mask[mask==0] = np.nan
                    for field in ['z0','dz']:
                        if field in temp.fields:
                            if mask.ndim == getattr(temp, field).ndim:
                                setattr(temp, field, getattr(temp, field) * mask)
                            else:
                                setattr(temp, field, getattr(temp, field) * mask[:,:,None])
                setattr(self, '_'+group, temp)
            except Exception as e:
                if VERBOSE:
                    print(f"surfaceModel.py: problem reading {this_file}")
                    print(e)
                pass

        return self

    def from_mosaic_tiles(self, directory, bounds=None, match_extent=False, include_lagrangian=True):

        if bounds is None:
            bounds=self.bounds()
        for group in ['z0','dz','lagrangian_dz']:
            try:
                print(group)
                this_dir = os.path.join(directory, group)
                print(this_dir)
                these_fields=[group, 'cell_area']
                if group=='lagrangian_dz':
                    these_fields=['dz']
                if not os.path.isdir(this_dir):
                    if group=='lagrangian_dz':
                        this_dir=os.path.join(directory, 'lag_dz.h5')
                        if not os.path.isdir(this_dir):
                            continue
                    else:
                        continue
                temp = pc.grid.mosaic().from_list(
                    glob.glob(os.path.join(this_dir, group+'*.h5')),
                            group=group, bounds=bounds,
                            fields=these_fields)
                if 'cell_area' in temp.fields:
                    mask=(temp.cell_area > 0.001*(temp.x[1]-temp.x[0])*(temp.y[1]-temp.y[0])).astype(float)
                    mask[mask==0] = np.nan
                    for field in ['z0','dz']:
                        if field in temp.fields:
                            setattr(temp, field, getattr(temp, field) * mask)
                setattr(self, '_'+group, temp)
            except Exception as e:
                print(e)
                pass



    def get_z(self, include_lagrangian=True):
        """
        Get the combined DEM elevation and elevation change

        Parameters
        ----------
        include_lagrangian : Boolean, optional
            If true, calculate the Lagrangian contribution. The default is True.
        """

        if self.shelf:
            # on the shelf, z0 and dz are both defined in Lagrangian coordinates.
            # Store the shifted (original-position) coordinates in x0, y0
            self._get_shifted_xy(assign=True)
            include_lagrangian=False

        if 'z0' not in self.fields:
            self.get_z0()
        if 'dz' not in self.fields:
            self.get_dz()

        self.assign(z=self.z0+self.dz)
        self.__update_size_and_shape__()
        self.__update_extent__()
        if include_lagrangian:
            self.get_lagrangian_dz()
            self.z += self.lagrangian_dz
        return self

    def get_z0(self):
        """
        Get the DEM elevation

        Returns
        -------
        None.

        """
        squeeze = np.isscalar(self.t)
        if self.shelf:
            self.assign(z0=self._z0.interp(self.x0, self.y0, field='z0'))
        else:
            self.assign(z0 = self._z0.interp(self.x, self.y, field='z0',  gridded=True))
        if not squeeze:
            self.z0=self.z0[:,:,None]


    def get_dz(self):
        squeeze = np.isscalar(self.t)
        if squeeze:
            t=[self.t]
        else:
            t=self.t
        if self.shelf:
            dz = self._dz.interp(self.x0, self.y0, t, field='dz')
        else:
            dz = self._dz.interp(self.x, self.y, t, field='dz', gridded=True)
        if squeeze:
            self.assign(dz=np.squeeze(dz))
        else:
            self.assign(dz=dz)
        return self

    def get_lagrangian_dz(self):

        squeeze = np.isscalar(self.t)
        if self._lagrangian_dz is not None:
            x0, y0 = self._get_shifted_xy()
            lag_dz =self._lagrangian_dz.interp(x0, y0, field='dz', gridded=False)
            if squeeze:
                self.assign(lagrangian_dz = np.squeeze(lag_dz))
            else:
                self.assign(lagrangian_dz = lag_dz)
        return self

    def get_z_ice(self,  rho_water=1.02, apply_FAC_anomaly = False):
        """
        Get the firn - corrected surface

        Parameters
        ----------
        rho_water : float, optional
            Water density. The default is 1.02.
        apply_FAC_anomaly : bool : optional
            If True, the FAC for the first time step in the firn model is
            subtracted.  The default is False.

        Returns
        -------
        surfaceModel
            updated surface model

        """

        squeeze = np.isscalar(self.t)
        if squeeze:
            t=[self.t]
        else:
            t=self.t

        SMB = pc.grid.data().from_file(self.SMB_file, bounds=self.bounds(pad=5.e4),
                                           t_range=[np.min(self.t)-0.5, np.max(self.t)+0.5],
                                            fields=['SMB_a'])
        SMB.SMB_a = pc.grid.fill_edges(SMB.SMB_a)
        SMB_a = SMB.interp(self.x, self.y, t, gridded=True, field='SMB_a')
        if 'floating' not in self.fields:
            if self.floating_mask_file is not None:
                self.assign(
                    floating = np.abs(pc.grid.data().from_file(self.floating_mask_file, bounds=self.bounds(pad=1.e3))\
                        .interp(self.x, self.y,gridded=True)
                        - self.floating_mask_value) < 0.01 )
        if 'floating' in self.fields:
            float_scale = ((self.floating==0) + (1-rho_water)/rho_water*(self.floating==1))[:,:,None]
        else:
            float_scale = np.ones(SMB_a.shape)

        if squeeze:
            print(self)
            self.assign(z_ice = self.z + np.squeeze(SMB_a*float_scale))
        else:
            self.assign(z_ice=self.z + SMB_a*float_scale)

        if apply_FAC_anomaly:
            temp=pc.grid.data().from_file(self.SMB_file, meta_only=True)
            t0 = temp.t[0]
            dt=temp.t[1]-temp.t[0]
            FAC0 =pc.grid.data().from_file(self.SMB_file, bounds=self.bounds(pad=5.e4), t_range=[t0, t0+dt], fields=['FAC'])
            FAC0.FAC = pc.grid.fill_edges(FAC0.FAC)
            dz_FAC0 = FAC0.interp(self.x, self.y, [t0], field='FAC', gridded=True)
            if squeeze:
                dz_FAC0 = np.squeeze(dz_FAC0)
            self.z_ice -= dz_FAC0
        return self

    def get_z_surf(self, calc_FAC_anomaly=True):
        squeeze = np.isscalar(self.t)
        if squeeze:
            t=[self.t]
        temp=pc.grid.data().from_file(self.SMB_file, meta_only=True)
        t0 = temp.t[0]
        dt = temp.t[1]-temp.t[0]
        t_range =[t[0]-2*dt, t[-1]+2*dt]

        FAC=pc.grid.data().from_file(self.SMB_file, bounds=self.bounds(pad=5.e4), t_range=t_range, fields=['FAC'])
        if calc_FAC_anomaly:
            FAC0 =pc.grid.data().from_file(self.SMB_file, bounds=self.bounds(pad=5.e4), t_range=[t0-dt, t0+dt], fields=['FAC'])
            FAC.FAC -= FAC0.FAC
        FAC.FAC=pc.grid.fill_edges(FAC.FAC)
        FAC_i = FAC.interp(self.x, self.y, t, field='FAC', gridded=True)
        if squeeze:
            self.assign(z_surf = self.z_ice + np.squeeze(FAC_i))
        else:
            self.assign(z_surf = self.z_ice + FAC_i)
        return self

    def _get_dz_ice(self, rho_water=1020):


        if 'dz' not in self.fields:
            self.get_dz()
        t_range = [self._dz.t[0]-0.1, self._dz.t[-1]+0.1]
        SMB=pc.grid.data().from_file(self.SMB_file,
                            bounds=self.bounds(pad=5.e4),
                            t_range=t_range,
                            fields=['SMB_a'])
        SMB.SMB_a = pc.grid.fill_edges(SMB.SMB_a)
        if 'floating' not in self.fields:
            if self.floating_mask_file is not None:
                self.assign(
                    floating = np.abs(
                        pc.grid.data().from_file(self.floating_mask_file, bounds=self.bounds(pad=1.e3))\
                            .interp(self.x, self.y,gridded=True)
                        - self.floating_mask_value) < 0.01 )
        if 'floating' in self.fields:
            floating = self.interp(self._dz.x, self._dz.y, gridded=True) > 0.5
            float_scale = ((floating==0) + (1-rho_water)/rho_water*(floating==1))[:,:,None]
        else:
            float_scale = 1

        self._dz.assign(SMB_a = float_scale*SMB.interp(self._dz.x, self._dz.y, self._dz.t, gridded=True, field='SMB_a'))
        self._dz.assign(dz_ice = self._dz.dz + self._dz.SMB_a)
