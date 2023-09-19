__all__ = ['get_file', 'wget_file', 'request_file', 'ftp_file']


def get_file(url, local_path, wget=False):
    """
    Download file from ftp or http via wget, ftp_file, or request_file

    Arguments
    ---------
    url : str
        Path on server
    local_path : str
        Path to save file (usually url without file protocol prefix
    wget : bool
        If True, use wget (default: False)

    Returns
    -------
    local_path : str
        local_path
    """
    if wget:
        return wget_file(url, local_path)
    elif url.startswith('ftp://'):
        return ftp_file(url, local_path)
    else:
        return request_file(url, local_path)


def ftp_file(url, local_path):
    """
    While files are on STAR ftp, use this function.

    Arguments
    ---------
    url : str
        Path on ftp server
    local_path : str
        Path to save file (usually url without ftp://)

    Returns
    -------
    local_path : str
        local_path
    """
    import ftplib
    import os

    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    server = url.split('//')[1].split('/')[0]
    remotepath = url.split(server)[1]
    ftp = ftplib.FTP(server)
    ftp.login()
    with open(local_path, 'wb') as fp:
        ftp.retrbinary(f'RETR {remotepath}', fp.write)
    ftp.quit()
    return local_path


def wget_file(url, local_path):
    """
    If local has wget, this can be used.

    Arguments
    ---------
    url : str
        Path on ftp server
    local_path : str
        Path to save file (usually url without ftp://)

    Returns
    -------
    local_path : str
        local_path
    """
    import os
    if not os.path.exists(local_path):
        cmd = f'wget -r -N {url}'
        os.system(cmd)

    return local_path


def request_file(url, local_path):
    """
    Only works with http and https

    Arguments
    ---------
    url : str
        Path on ftp server
    local_path : str
        Path to save file (usually url without https://)

    Returns
    -------
    local_path : str
        local_path
    """
    import requests
    import shutil
    import os

    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_path


def read_netrc(netrcpath, server):
    import netrc
    nf = netrc.netrc(netrcpath)
    return nf.authenticators(server)
