from configparser import  ConfigParser

class OrdinaryKriging(object):
    def __init__(self):

        return
    def _load_configs(self):
        config = ConfigParser()
        config.read('odrinary_kriging.ini')
        if 'y' in config['generate_pathloss_from_log_normal']:
            self._generate_lognormal_pathloss_data()
        return
    def _generate_lognormal_pathloss_data(self):

        return

    def ruan_app(self):

        return
if __name__ == '__main__':
    ok = OrdinaryKriging()
    ok.run_app()