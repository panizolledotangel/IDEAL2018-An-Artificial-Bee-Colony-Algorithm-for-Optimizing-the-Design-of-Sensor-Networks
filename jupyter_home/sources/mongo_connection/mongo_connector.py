"""
    This static class stores the connection to a MongoDB database so it can be used by all controllers and be initialize
    out of them, like this the same code of the controllers can be used for testing and for production.
"""
import pymongo


class MongoDBConnection(object):
    """
    the connection object, this object should be private
    """
    host = None
    port = None
    driver = None

    @classmethod
    def initialize_connection(cls, host: str, port: int):
        """
        initialize the class with a new connection
        :param host: url where the MongoDB is stored
        :param port: port where the MongoDB is listening
        :return None
        """
        cls.host = host
        cls.port = port
        cls.driver = pymongo.MongoClient(cls.host, cls.port,
                                         username='angel',
                                         password='rootPassXXX')

    @classmethod
    def reinitialize(cls):
        """
        Call this method after a fork is done to reinitialize the MongoClient. This method does not need to be called if
        a new thread is created only if a fork is created.
        :return:
        """
        cls.driver = pymongo.MongoClient(cls.host, cls.port,
                                         username='angel',
                                         password='rootPassXXX')

    @classmethod
    def get_iterations_db(cls):
        """
        Gets the actual connection to the MongoDB database, and exception is raised if the connection is not initialized
        :return: the connection object to the MongoDB database
        """
        if cls.driver is None:
            raise RuntimeError("Database Connection need to be initialised")
        else:
            return cls.driver.experiments_db.iterations
