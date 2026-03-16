def file_upload(filename,host,token):
    """ Upload file filename to host host using authorization token token.
    If successful, returns the URL for the publicly visible uploaded data, else
    raises RuntimeError.
    """
    with open(filename,'rb') as infile:
        r=requests.post(host,data={'auth':token},files={'file': infile})
    status=r.json()['status']
    if status!='success':
        raise RuntimeError(f'Upload failed with status {status}')
    else:
        return r.json()['url']
