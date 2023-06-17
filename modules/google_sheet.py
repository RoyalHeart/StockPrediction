from __future__ import print_function

import os

import httplib2
from apiclient import discovery


def main(key="AIzaSyCugg23-6t0EBF3Eqp5vYNDTxjpvzsrSck"):
    discoveryUrl = "https://sheets.googleapis.com/$discovery/rest?version=v4"
    service = discovery.build(
        "sheets",
        "v4",
        http=httplib2.Http(),
        discoveryServiceUrl=discoveryUrl,
        developerKey=key,
    )

    spreadsheetId = "1u0jaxxeZzmB48CIeOZlSHPMGWPSkO6enR6LPwDgCUC4"
    rangeName = "Stock!A1:C"
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheetId, range=rangeName)
        .execute()
    )
    values = result.get("values", [])

    if not values:
        print("No data found.")
    else:
        print("Name, Major:")
        for row in values:
            # Print columns A and E, which correspond to indices 0 and 4.
            print("%s" % (row[0]))


if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        main(key=argv[1])
    else:
        main()
