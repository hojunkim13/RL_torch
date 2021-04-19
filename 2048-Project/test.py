values = [0,0,0,0]    
for first_action in range(4):
    for sim in range(10000 // 4):
        value = 0
        while value != 100:
            value += 1
        #value = self.calcValue(grid)
        values[first_action] += value

print(values)