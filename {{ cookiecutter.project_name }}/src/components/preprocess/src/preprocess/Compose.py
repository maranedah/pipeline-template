class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, df):
        for t in self.transforms:
            if len(df):
                df = t(df)
        return df
