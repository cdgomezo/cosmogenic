import os
import datetime
import numpy as np
import netCDF4 as nc


def month_start_days(year):
    """Day-of-year offsets of the 12 month starts relative to {year}-01-01.

    Monthly fluxes are stamped at the START of the month; this is the
    convention used throughout, so that files can be combined on a common time
    axis without alignment surprises.

    Computed rather than tabulated so leap years are correct: 1 March is day
    59 in a common year and day 60 in a leap year.
    """
    jan1 = datetime.date(year, 1, 1)
    return [float((datetime.date(year, m, 1) - jan1).days) for m in range(1, 13)]


def write_year(year, data_3d, lat, lon, aux, config):
    """
    Write one year of columnar C_D14C flux to a NetCDF4 file.

    The field is (time, lat, lon) with NO vertical dimension: TM5 performs the
    vertical distribution itself from the live meteorological fields, so this
    package ships the column-integrated flux only.

    year     : calendar year (int)
    data_3d  : flux [umol C_D14C m^-2 s^-1], shape (12, nlat, nlon)
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
    path = os.path.join(out_dir, f"flux_c14.cosmogenic.{year}.nc")

    _, nlat, nlon = data_3d.shape

    zlib    = compress
    clevel  = 4 if compress else 0

    with nc.Dataset(path, 'w', format='NETCDF4') as ds:
        # Dimensions — sized from data_3d so they are always consistent
        ds.createDimension('time', 12)
        ds.createDimension('lat',  nlat)
        ds.createDimension('lon',  nlon)

        # Coordinate: time (month starts as days since epoch)
        t_var = ds.createVariable('time', 'f8', ('time',))
        t_var.units         = f'days since {year}-01-01 00:00:00'
        t_var.calendar      = 'standard'
        t_var.long_name     = 'time'
        t_var.standard_name = 'time'
        t_var.axis          = 'T'
        t_var[:] = np.array(month_start_days(year), dtype='f8')

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
        c14 = ds.createVariable('c14flux', 'f4',
                                ('time', 'lat', 'lon'),
                                zlib=zlib, complevel=clevel)
        c14.units     = 'umol m-2 s-1'
        c14.long_name = 'Cosmogenic 14C production as C_D14C flux'
        c14.comment   = (
            'C_D14C = CO2 x Delta14C with Delta14C dimensionless (permil/1000), '
            'following Basu et al. (2016). Column-integrated: TM5 distributes '
            'this in the vertical from the meteorological fields.'
        )
        c14[:] = data_3d.astype('f4')

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
        ds.title          = 'Cosmogenic 14C production as C_D14C flux for TM5'
        ds.year           = str(year)
        ds.vertical_grid  = 'none — column field; TM5 distributes vertically'
        ds.geomag_model   = 'IGRF via ppigrf'
        ds.created        = datetime.datetime.now(datetime.timezone.utc).isoformat()
