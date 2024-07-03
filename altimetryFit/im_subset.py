from osgeo import gdal
import numpy as np
#from altimetryFit import check_obj_memory


class im_subset:
    def __init__(self, c0, r0, Nc, Nr, source, pad_val=0, Bands=(1,2,3), stride=None, pad=None, no_edges=False):
        self.source=source
        self.c0=c0
        self.r0=r0
        self.Nc=Nc
        self.Nr=Nr
        self.z=[]
        if hasattr(self.source, 'level'): # if the level is zero, this is a copy of a file, if it's >0, it's a copy of a copy of a file
            self.level=self.source.level+1
        else:
            self.level=0
        self.Bands=Bands
        self.pad_val=pad_val
        if stride is not None:
            if not hasattr(stride,"__len__"):
                stride=np.array([stride, stride])
            if pad is None:
                pad=0
            if not hasattr(pad,"__len__"):
                pad=np.array([pad, pad])
            self.stride=stride
            self.pad=pad
            if not no_edges:
                x0=self.c0+np.arange(0, self.Nc, stride[0])
                y0=self.r0+np.arange(0, self.Nr, stride[1])
            else:
                x0=self.c0+np.arange(pad[0], self.Nc-pad[0], stride[0])
                y0=self.r0+np.arange(pad[1], self.Nr-pad[1], stride[1])
            [x0,y0]=np.meshgrid(x0, y0)
            self.xy0=np.c_[x0.ravel(), y0.ravel()]
            self.count=0

    def __getitem__(self, index):
        self.setBounds( self.xy0[index,0]-self.pad[0], self.xy0[index,1]-self.pad[1],
                              self.stride[0]+2*self.pad[0], self.stride[1]+2*self.pad[1])
        self.copySubsetFrom()
        return self

#    def __iter__(self):
#        return self
#
#    def __next__(self):
#        if self.count < self.xy0.shape[0]:
#            self.setBounds( self.xy0[self.count,0]-self.pad[0], self.xy0[self.count,1]-self.pad[1],
#                              self.stride[0]+2*self.pad[0], self.stride[1]+2*self.pad[1])
#            self.count=self.count+1
#            self.copySubsetFrom()
#            return self
#        else:
#            raise StopIteration
#    def __next__(self):

    def setBounds(self, c0, r0, Nc, Nr, update=0):
        self.c0=int(c0)
        self.r0=int(r0)
        self.Nc=int(Nc)
        self.Nr=int(Nr)
        if update > 0:
            self.copySubsetFrom(pad_val=self.pad_val)

    def copySubsetFrom(self, pad_val=0):
        datatype_dict={
                'Byte':np.ubyte, \
                'Float32':np.float32, 'Float64':np.float64,\
                'Int16':np.int16,\
                'Int32':np.int32,'Uint32':np.uint32,\
                'Int64':np.int32,'Uint64':np.uint64}
        if hasattr(self.source, 'level'):  # copy data from another subset
            self.z = np.zeros((self.source.z.shape[0], self.Nr, self.Nc), self.source.z.dtype) + pad_val
            (sr0, sr1, dr0, dr1, vr)=match_range(self.source.r0, self.source.Nr, self.r0, self.Nr)
            (sc0, sc1, dc0, dc1, vc)=match_range(self.source.c0, self.source.Nc, self.c0, self.Nc)
            if (vr & vc):
                self.z[:, dr0:dr1, dc0:dc1]=self.source.z[:,sr0:sr1, sc0:sc1]
            self.level=self.source.level+1
        else:  # read data from a file
            band=self.source.GetRasterBand(int(self.Bands[0]))
            src_NB=self.source.RasterCount
            dt=datatype_dict[gdal.GetDataTypeName(band.DataType)]
            try:
                assert(
                    (self.z.dtype==dt) &
                    np.all(self.z.shape==np.array([src_NB, self.Nr, self.Nc])))
                self.z[:]=pad_val
            except Exception:
                self.z=np.zeros((src_NB, self.Nr, self.Nc), dt)+pad_val
            (sr0, sr1, dr0, dr1, vr)=match_range(0, band.YSize, self.r0, self.Nr)
            (sc0, sc1, dc0, dc1, vc)=match_range(0, band.XSize, self.c0, self.Nc)
            if (vr & vc):
                a=self.source.ReadAsArray(int(sc0),  int(sr0), int(sc1-sc0), int(sr1-sr0))
                self.z[:, dr0:dr1, dc0:dc1]=a
            self.level=0


    def writeSubsetTo(self, bands, target):
        if hasattr(target, 'level') and target.level > 0:
            print("copying into target raster")
            (sr0, sr1, dr0, dr1, vr)=match_range(target.source.r0, target.source.Nr, self.r0, self.Nr)
            (sc0, sc1, dc0, dc1, vc)=match_range(target.source.c0, target.source.Nc, self.c0, self.Nc)
            if (vr & vc):
                for b in bands:
                    target.source.z[b,sr0:sr1, sc0:sc1]=self.z[b, dr0:dr1, dc0:dc1]
        else:
#            print("writing to file");
            band=target.source.GetRasterBand(1)
            (sr0, sr1, dr0, dr1, vr)=match_range(0, band.YSize, self.r0, self.Nr)
            (sc0, sc1, dc0, dc1, vc)=match_range(0, band.XSize, self.c0, self.Nc)
#            print (sc0, sc1, dc0, dc1)
#            print (sr0, sr1, dr0, dr1)
#            print "vr=", vr, "vc=", vc
            if (vr & vc):
#                print "...writing..."
                try:
                    for bb in (bands):
                        band=target.source.GetRasterBand(int(bb))
                        band.WriteArray( self.z[bb-1, dr0:dr1, dc0:dc1], int(sc0), int(sr0))
                except TypeError:
                     band=target.source.GetRasterBand(int(bands))
                     band.WriteArray( self.z[int(bands-1), dr0:dr1, dc0:dc1], int(sc0), int(sr0))

    def iterateFrom(self, stride, pad=0, no_edges=False):
        #        use this to loop over sub images of an image
        # for im in im_sub.iterateFrom(stride, pad)
        if not hasattr(stride,"__len__"):
            stride=np.array([stride, stride])
        if not hasattr(pad,"__len__"):
            pad=np.array([pad, pad])
        if no_edges:
            x0=np.arange(pad, self.Nc-pad, stride[0])
            y0=np.arange(pad, self.Nr-pad, stride[1])
        else:
            x0=np.arange(0, self.Nc, stride[0])
            y0=np.arange(0, self.Nr, stride[1])
        for x0i in x0:
            for y0i in y0:
                new_sub=im_subset( x0i-pad[0], y0i-pad[1], stride[0]+2*pad[0], stride[1]+2*pad[1], self.source, pad_val=self.pad_val, Bands=self.Bands)
                new_sub.copySubsetFrom()
                yield new_sub

    #def get_memory_usage(self):
    #    return check_obj_memory(self)

def match_range(s0, ns, d0, nd):
    i0 = max(s0, d0)
    i1 = min(s0+ns, d0+nd)
    si0=max(0, i0-s0)
    si1=min(ns, i1-s0)
    di0=max(0, i0-d0)
    di1=min(nd,i1-d0)
    any_valid=(di1>di0) & (si1 > si0)
    return (si0, si1, di0, di1, any_valid)
