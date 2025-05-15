import sys

class WriteWrapper:
    def __init__(self, path):
        self.path = path
        if (self.path == None): return
        file = open(path, 'w')
        file.write(f"Writing to {path}\n")
        file.close()
    
    def __call__(self, data):
        if (self.path == None):
            print(data)
        else:
            file = open(self.path, 'a')
            file.write(f"{data}\n")
            file.close()

    def __call__(self, *data):
        if (self.path == None):
            print(' '.join([str(d) for d in data]))
        else:
            file = open(self.path, 'a')
            file.write(f"{' '.join([str(d) for d in data])}\n")
            file.close()
        
    def Important(self, data):
        print(data)
        if (self.path != None):
            file = open(self.path, 'a')
            file.write(f"{data}\n")
            file.close()
    
    def Important(self, *data):
        print(' '.join([str(d) for d in data]))
        if (self.path != None):
            file = open(self.path, 'a')
            file.write(f"{' '.join([str(d) for d in data])}\n")
            file.close()

    def flush(self): pass
    
    def close(self): pass