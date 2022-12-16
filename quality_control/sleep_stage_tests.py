import math


#class RcsLabelTests(unittest.TestCase):
class RcsLabelTests:
    def __init__(self, rcs_df, dreem_report_dict, sample_rate):
        self.df = rcs_df
        self.report = dreem_report_dict
        self.sr = sample_rate
        self.SECONDS_PER_MINUTE = 60
        self.set_up()
        self.test_n3()
        self.test_n2()
        self.test_n1()
        self.test_rem()

    def set_up(self):
        self.N3 = round(len(self.df[self.df["SleepStage"] == 2]) / self.sr / self.SECONDS_PER_MINUTE)
        self.N2 = round(len(self.df[self.df["SleepStage"] == 3]) / self.sr / self.SECONDS_PER_MINUTE)
        self.N1 = round(len(self.df[self.df["SleepStage"] == 4]) / self.sr / self.SECONDS_PER_MINUTE)
        self.REM = round(len(self.df[self.df["SleepStage"] == 5]) / self.sr / self.SECONDS_PER_MINUTE)

    def test_n3(self):
        dreem_n3 = round(float(self.report['n3_duration']))
        try:
            assert self.N3 == dreem_n3  # add assertion here
            print("\t\tPASSED: N3 Duration Test")
        except AssertionError:
            print(f"\t\tFAILED: N3 Duration Test. Parquet N3 duration ({self.N3} min). Dreem N3 duration ({dreem_n3} min)")

    def test_n2(self):
        dreem_n2 = round(float(self.report['n2_duration']))
        try:
            assert self.N2 == dreem_n2  # add assertion here
            print("\t\tPASSED: N2 Duration Test")
        except AssertionError:
            print(f"\t\tFAILED: N2 Duration Test. Parquet N2 duration ({self.N2} min). Dreem N2 duration ({dreem_n2} min)")

    def test_n1(self):
        dreem_n1 = round(float(self.report['n1_duration']))
        try:
            assert self.N1 == dreem_n1  # add assertion here
            print("\t\tPASSED: N1 Duration Test")
        except AssertionError:
            print(f"\t\tFAILED: N1 Duration Test. Parquet N1 duration ({self.N1} min). Dreem N1 duration ({dreem_n1} min)")

    def test_rem(self):
        dreem_rem = round(float(self.report['rem_duration']))
        try:
            assert self.REM == dreem_rem  # add assertion here
            print("\t\tPASSED: REM Duration Test")
        except AssertionError:
            print(f"\t\tFAILED: REM Duration Test. Parquet REM duration ({self.REM} min). Dreem REM duration ({dreem_rem} min)")


# if __name__ == '__main__':
#     unittest.main()
