from src.objective_function import ObjectiveFunction

def test_obj_get_score():
    obj = ObjectiveFunction()
    score = obj.get_score(debug=True)
    print(score)


def test_get_placements():
   from src import objective_function
   import numpy as np

   xgrid = np.arange(-5, 5+0.5, 0.5)
   ygrid = np.arange(0, 5+0.5, 0.5)

   slots = objective_function.get_slots_from_side(xgrid, 3)
   print(slots)


def main():
    #test_obj_get_score()
    test_get_placements()


if __name__ == '__main__':
    main()