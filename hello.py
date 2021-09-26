arr=[-3, 1, -3, 3, 3, -1, 3, -3, 3, 1, -3, 3, 3, 1, 1, 1, -1, -3, -3, -3, 1, 3, -3, 3, -1, 1, 3, -1, 3, -3, 1, 1]

with open('output.txt', 'w') as f:
    for item in arr:
        if item>0:
            f.write("+1\n")
        else:
            f.write("-1\n")