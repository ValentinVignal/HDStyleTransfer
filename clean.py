from epicpath import EPath

for path in ['content', 'style', 'results']:
    EPath(path).rm()


