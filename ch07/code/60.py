train_cnt1 = 0
train_cnt0 = 0
dev_cnt1 = 0
dev_cnt0 = 0

with open("data/SST-2/train.tsv","r",encoding="utf-8") as f:
    for i in f:
        line = i.strip().split()
        print(line)
        label = line[-1]
        if label == "1":
            train_cnt1+= 1
        elif label == "0":
            train_cnt0+=1
            
    print("----- train -----")
        
    print(f"positive(1) :{train_cnt1}")
    print(f"negative(0) :{train_cnt0}")


with open("data/SST-2/dev.tsv","r") as f:
    for i in f:
        line = i.strip().split()
        label = line[-1]
        if label == "1":
            dev_cnt1+= 1
        elif label == "0":
            dev_cnt0+=1
    
    print("----- dev -----")
        
    print(f"positive(1) :{dev_cnt1}")
    print(f"negative(0) :{dev_cnt0}")