def sum(n):
    if n == 0:
        return 0
    else:
        return sum(n - 1) + n


def findPath(n, m):
    if 1 in (n, m):
        return 1
    else:
        return findPath(n - 1, m) + findPath(n, m-1)


