# Standard Library
import pickle
import logging

logger = logging.getLogger("Delorean Helper")


class Helper:
    def __init__(self):
        pass

    def serialize(self):
        """
        Serialize the Helper instance using pickle.

        :return: A byte stream representing the serialized Helper instance.
        """
        return pickle.dumps(self)

    @staticmethod
    def deserialize(serialized_data):
        """
        Deserialize a Helper instance from a byte stream.
        This is a static method that can be called without creating an instance of the class.

        :param serialized_data: A byte stream representing the serialized Helper instance.
        :return: A deserialized Helper instance.
        """
        return pickle.loads(serialized_data)
