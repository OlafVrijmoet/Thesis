
def get_yes_no_input(prompt):
    while True:
        user_input = input(prompt).lower().strip()
        if user_input in ('yes', 'y', 'no', 'n'):
            return user_input in ('yes', 'y')
        else:
            print("Invalid input. Please enter 'Yes' or 'No'.")
