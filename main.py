import yaml


if __name__ == "__main__":
    with open("config.yaml", "r") as configfile:
        config_dict = yaml.load(configfile, Loader=yaml.FullLoader)
    print(config_dict)
