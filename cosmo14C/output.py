import os
import datetime
import numpy as np
import netCDF4 as nc


def write_year(year, data_4d, lat, lon, aux, config):
    """
    Write one year of 3D isoflux data to a NetCDF4 file.

    year     : calendar year (int)
    data_4d  : isoflux [TgC permil yr^-1], shape (12, nlev, nlat, nlon)
    lat      : latitude cell centres [degrees], shape (nlat,)
    lon      : longitude cell centres [degrees], shape (nlon,)
    aux      : dict with keys:
                 phi        : shape (12,) [MV]
                 Pc         : shape (nlat, nlon) [GV]
                 q_col      : shape (12, nlat, nlon) [atoms cm^-2 s^-1]
                 geomag_lat : shape (nlat, nlon) [degrees]
    config   : dict from run.yaml (used for output.directory, output.compress)
    """
    out_dir  = config['output']['directory']
    compress = config['output'].get('compress', True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cosmo14C_{year}.nc")

    _, nlev, nlat, nlon = data_4d.shape

    zlib    = compress
    clevel  = 4 if compress else 0

    with nc.Dataset(path, 'w', format='NETCDF4') as ds:
        # Dimensions — sized from data_4d so they are always consistent
        ds.createDimension('time', 12)
        ds.createDimension('lev',  nlev)
        ds.createDimension('lat',  nlat)
        ds.createDimension('lon',  nlon)

        # Coordinate: time (month midpoints as days since epoch)
        t_var = ds.createVariable('time', 'f8', ('time',))
        t_var.units         = f'days since {year}-01-01 00:00:00'
        t_var.calendar      = 'standard'
        t_var.long_name     = 'time'
        t_var.standard_name = 'time'
        t_var.axis          = 'T'
        # Month midpoints: day 15 of each month (approximate)
        month_middays = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
        t_var[:] = np.array(month_middays, dtype='f8')

        # Coordinate: lev
        lev_var = ds.createVariable('lev', 'i4', ('lev',))
        lev_var.long_name = 'TM5 tropo34 layer index (1=surface, nlev=top)'
        lev_var.units     = "1"
        lev_var.axis      = "Z"
        lev_var.positive  = "down"  # level 1 = surface, index increases toward TOA
        lev_var[:] = np.arange(1, nlev + 1)

        # Coordinate: lat
        lat_var = ds.createVariable('lat', 'f4', ('lat',))
        lat_var.units         = 'degrees_north'
        lat_var.long_name     = 'latitude'
        lat_var.standard_name = 'latitude'
        lat_coords = np.asarray(lat, dtype='f4')
        if len(lat_coords) == nlat:
            lat_var[:] = lat_coords
        else:
            # Fallback: fill with NaN so dimension is still sized correctly
            lat_var[:] = np.full(nlat, np.nan, dtype='f4')

        # Coordinate: lon
        lon_var = ds.createVariable('lon', 'f4', ('lon',))
        lon_var.units         = 'degrees_east'
        lon_var.long_name     = 'longitude'
        lon_var.standard_name = 'longitude'
        lon_coords = np.asarray(lon, dtype='f4')
        if len(lon_coords) == nlon:
            lon_var[:] = lon_coords
        else:
            lon_var[:] = np.full(nlon, np.nan, dtype='f4')

        # Primary variable
        c14 = ds.createVariable('C14_isoflux', 'f4',
                                ('time', 'lev', 'lat', 'lon'),
                                zlib=zlib, complevel=clevel)
        c14.units     = 'TgC permil yr-1'
        c14.long_name = 'Cosmogenic 14C isoflux'
        c14[:] = data_4d.astype('f4')

        # Diagnostics
        phi_v = ds.createVariable('phi', 'f4', ('time',))
        phi_v.units    = 'MV'
        phi_v.long_name = 'Solar modulation potential'
        phi_v[:] = aux['phi'].astype('f4')

        Pc_v = ds.createVariable('Pc', 'f4', ('lat', 'lon'))
        Pc_v.units    = 'GV'
        Pc_v.long_name = 'Vertical geomagnetic cutoff rigidity'
        Pc_arr = aux['Pc'].astype('f4')
        if Pc_arr.shape == (nlat, nlon):
            Pc_v[:] = Pc_arr
        else:
            Pc_v[:] = np.full((nlat, nlon), np.nan, dtype='f4')

        qcol_v = ds.createVariable('q_col', 'f4', ('time', 'lat', 'lon'),
                                   zlib=zlib, complevel=clevel)
        qcol_v.units    = 'atoms cm-2 s-1'
        qcol_v.long_name = 'Columnar 14C production rate'
        qcol_arr = aux['q_col'].astype('f4')
        if qcol_arr.shape == (12, nlat, nlon):
            qcol_v[:] = qcol_arr
        else:
            qcol_v[:] = np.full((12, nlat, nlon), np.nan, dtype='f4')

        glat_v = ds.createVariable('geomag_lat', 'f4', ('lat', 'lon'))
        glat_v.units    = 'degrees'
        glat_v.long_name = 'Geomagnetic latitude'
        glat_arr = aux['geomag_lat'].astype('f4')
        if glat_arr.shape == (nlat, nlon):
            glat_v[:] = glat_arr
        else:
            glat_v[:] = np.full((nlat, nlon), np.nan, dtype='f4')

        # Global attributes
        ds.title          = 'Cosmogenic 14C isoflux for TM5'
        ds.year           = str(year)
        ds.vertical_grid  = 'TM5 tropo34 (34 ECMWF L137 hybrid layers)'
        ds.vertical_shape = 'Masarik & Beer (2009) Table 1, normalized'
        ds.geomag_model   = 'IGRF via ppigrf'
        ds.created        = datetime.datetime.now(datetime.timezone.utc).isoformat()
