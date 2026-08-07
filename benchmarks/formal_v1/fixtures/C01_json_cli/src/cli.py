import sys

def main():
    for line in sys.stdin:
        if line.strip():
            print(line.strip())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
