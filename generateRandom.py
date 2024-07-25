from BlumBlumShub import BBSrandom

seed = 11
if __name__ == "__main__":
    numbers = int(input("Enter the number of numbers you want to generate: "))
    rand = BBSrandom()
    for i in range(0, numbers):
        print(rand.next())
