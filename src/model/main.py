from model.base import Model
from utils.serializer import serialize_rainfall_data

def main():
    m : Model = Model()
    serialize_rainfall_data(m.results, "results.json", summary_graph=m.summary)

if __name__ == '__main__':
    main()
