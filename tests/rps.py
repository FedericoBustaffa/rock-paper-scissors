from grps import RPSModel

if __name__ == "__main__":
    model = RPSModel(10, 10, rng=42)
    model.run_for(1)
