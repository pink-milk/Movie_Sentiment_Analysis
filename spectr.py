def passw(lenn,start,words):
    letters=['a','b','c','d','e','f','g','h','i','j','k','l','m']
    if(lenn==0):
        return
    for i in range(len(letters[start:])):
        if i in words:
            words[i]=words[i]+letters[i]
        else:
            words[i]=letters[i]
            # print(words[i])
            # print(words[i]+letters[i])
        lenn-=1
        start+=1
        passw(lenn,start,words)

        print(words)

def generate(arr, i, s, len, keep):
    
    # base case
    if (i == 0): # when len has
                 # been reached
        
        keep.append(s)
        return keep
     
    # iterate through the array
    for j in range(0, len):
 
        # Create new string with
        # next character Call
        # generate again until
        # string has reached its len
        appended = s + arr[j]
        generate(arr, i - 1, appended, len, keep)
 
    return keep
 
# function to generate
# all possible passwords
def crack(arr, len):
    keep=[]
    # call for all required lengths
    for i in range(1 , len + 1):
        generate(arr, i, "", len, keep)
    return keep

def check(keep,leng):
    ret=[]
    for i in range(len(keep)):
        if(len(keep[i])==leng):
            s=sorted(keep[i])
            
            # print(keep[i])
            a=''.join(s)
            if a not in ret:
                ret.append(a)
    return ret

arr1=['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
leng = len(arr1)
keep=crack(arr1, leng)

print(check(keep,4))

# words={}
# print(passw(4,0,words))