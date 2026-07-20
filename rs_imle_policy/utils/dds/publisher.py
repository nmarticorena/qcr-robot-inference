from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


def publish_string(channel: str, data: str) -> None:
    """Publish one string message to an initialized DDS channel."""
    publisher = ChannelPublisher(channel, String_)
    publisher.Init()
    publisher.Write(String_(data=data))
