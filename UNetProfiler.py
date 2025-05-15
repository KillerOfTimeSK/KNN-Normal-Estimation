import torch
import time

class UNetProfiler():
    def __init__(self, file):
        self.file = file
        self.StartTime = time.time()
        self.eventList = []
        self.eventList.append(("Created", time.time() - self.StartTime, torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024, None))

    def __call__(self, name, shape, autoPrint=False):
        eventTime = time.time() - self.StartTime
        memory = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
        self.eventList.append((name, eventTime, memory, shape))
        missingSpaces = 30 - len(name)
        name = name + " " * (missingSpaces // 2) + " " * (missingSpaces - missingSpaces // 2)
        if autoPrint: self.file(f"Event name: {name} - Time: {eventTime:.0f}s, Memory: {memory:.1f}GB, Shape: {shape}")

    def printEvents(self):
        for i in range(len(self.eventList)):
            name, eventTime, memory, shape = self.eventList[i]
            if i == 0:
                self.file(f"Event name: {name} - Time: {eventTime:.0f}s, Memory: {memory:.1f}GB")
            else:
                prevEventTime, prevMemory = self.eventList[i - 1][1:3]
                timeDiff = eventTime - prevEventTime
                memoryDiff = memory - prevMemory
                self.file(f"Event name: {name} - Time: {timeDiff:.0f}s, Memory: {memoryDiff:.1f}GB, Shape: {shape}")