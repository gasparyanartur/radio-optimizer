from src.objective_function import ObjectiveFunction

def main():
    obj = ObjectiveFunction()
    score = obj.get_score(debug=True)
    print(score)

if __name__ == '__main__':
    main()