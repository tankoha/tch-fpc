Write a Free Pascal program using tch-fpc and `TchNN` that creates a
linear layer 4 -> 2 in one variable store, saves the store to a file in
the system's temporary directory, loads that file into a second store
holding a linear layer 4 -> 2 under the same name, and checks that both
layers give the same output for an input of ones. Delete the file at the
end. Print exactly:

    same = TRUE
