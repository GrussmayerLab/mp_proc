# metadata parsing for micromanager 2.x in KG lab
import re 
import yaml

def openMetadata(filepath):
    text = []
    with open(filepath) as f:
        text = f.readlines()
    return text


def getHeader(text):
    end_condition = r"\"FrameKey-\d+-\d+-\d+\"\:\s\{"
    pattern = re.compile(r"\"(.*?)\"\:\s(.*?|\"(.*?)\")\,")
    header = {}
    for i, line in enumerate(text):
        if re.match(end_condition, line):
            return header
            break # not sure if this is needed or if python returns automatically but lets just keep it as exid condition I guess
        else:
            for match in re.finditer(pattern, line):
                key = match.group(1)
                info = match.group(2)
                header[key] = info
    return header
            

def getFrames(text):
    REG = re.compile("(FrameKey-)(\d+-\d+-\d+)")
    frames = {} 
    raw_text = []
    F = 0 
    for i, line in enumerate(text):
        for match in re.finditer(REG, line):
            if F == 0:
                key = match.group(2)
                frames[F] = {"key": key, "text": []} 
                raw_text = []
            else: 
                frames[F-1]["text"] = raw_text[1:]     
                key = match.group(2)
                frames[F] = {"key": key, "text": []}
                raw_text = []
            F += 1    
        raw_text.append(line)    
    frames[F-1]["text"] = raw_text[1:]   
    #print('Found on line %s: %s' % (i+1, match.group()))

    # parse the frame info from text into a dictionary 
    for idx, frame in frames.items():
        frames[idx]["info"] = parse_single_frame_info(frame["text"])
        del frames[idx]['text'] 

    return frames


def parse_single_frame_info(text):
    info_dict = {}
    for line in text:
        # Strip leading/trailing whitespace and remove commas and quotes
        line = line.strip().rstrip(',')
        if ':' in line:
            key_part, value_part = line.split(':', 1)
            key = key_part.strip().strip('"')
            value = value_part.strip().strip('"')
            info_dict[key] = value
    return info_dict

    
def getXPosFromFrame(metadata, frame):
    FRAMES = []
    return metadata


def parseTextperFrame(FrameDict): 
    pattern = re.compile(r"\"(.*?)\"\:\s(.*?|\"(.*?)\")\,")
    for frame in FrameDict: 
        text = FrameDict[frame]["text"]
        for line in text: 
            for match in re.finditer(pattern, line):
                FrameDict[frame][match.group(1)] = match.group(2)
                
        del FrameDict[frame]["text"]
        
    return FrameDict


def getTimeStamps(FrameDict):
    text = []
    for frame in FrameDict: 
        text.append(float(FrameDict[frame]["ElapsedTime-ms"]))
    return text
        
        
def parseILAS(text): 
    framedict = {"frames_": {}}
    header = re.compile(r'\"(.*?)\"\,\s\"(.*?)\"')
    frames = re.compile(r'(\d*?)\,\s\"(\d*?\.\d*?)\"')
    for line in text: 
        for match in re.finditer(header, line):
            framedict[match.group(1)] = match.group(2)
        for match in re.finditer(frames, line):
            if match.group(1) is not None: 
                framedict["frames_"][int(match.group(1))] = float(match.group(2))
    return framedict

def timecodeILAS(framedict):
    timecode = []
    for key, time in framedict["frames_"].items():
        timecode.append(float(time*1000))
    return timecode


def addPicassoInfo(framedict):
    framedict["Data Type"] = "uint16"
    framedict["Height"] = int(framedict["Image height"])
    framedict["Width"] = int(framedict["Image width"])
    framedict["Frames"] = max(framedict["frames_"].keys())
    framedict["Byte Order"] = "<"
    return framedict

def writeYAML_ILAS(framedict, filepath): 
    framedict = addPicassoInfo(framedict)
    with open(filepath, 'w') as yaml_file:
        yaml.dump(framedict, yaml_file, default_flow_style=False)   
    return framedict

def writeYAML(framedict, filepath): 
    with open(filepath, 'w') as yaml_file:
        yaml.dump(framedict, yaml_file, default_flow_style=False)   


def readYAML(filepath):
    with open(filepath, 'r') as file:
        framedict = yaml.safe_load(file)
    return framedict
    