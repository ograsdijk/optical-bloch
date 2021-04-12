def recursive_subscript(i):
    # chr(0x2080+i) is unicode for
    # subscript num(i), resulting in x₀₀ for example
    if i < 10:
        return chr(0x2080+i)
    else:
        return recursive_subscript(i//10)+chr(0x2080+i%10)