def excel_col(col):
    ''' convert 1-relative column number to excel-style column label '''
    quot, rem = divmod(col - 1, 26)
    return excel_col(quot) + chr(rem + ord('A')) if col != 0 else ''
