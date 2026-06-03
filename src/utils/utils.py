# ----------------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------------

from typing import NamedTuple


# ----------------------------------------------------------------------------
#  Formatter
# ----------------------------------------------------------------------------

class ColorFormat(NamedTuple):
    name: str
    ansi: str


class Formatter:
    BLUE = ColorFormat("blue", "\033[94m")
    BROWN = ColorFormat("brown", "\033[38;5;130m")
    CRIMSON = ColorFormat("crimson", "\033[38;5;161m")
    CYAN = ColorFormat("cyan", "\033[96m")
    DARKRED = ColorFormat("darkred", "\033[38;5;52m")
    GOLD = ColorFormat("gold", "\033[38;5;220m")
    GRAY = ColorFormat("gray", "\033[90m")
    GREEN = ColorFormat("green", "\033[92m")
    LIME = ColorFormat("lime", "\033[38;5;118m")
    MAGENTA = ColorFormat("magenta", "\033[95m")
    MAROON = ColorFormat("maroon", "\033[38;5;88m")
    ORANGE = ColorFormat("orange", "\033[38;5;208m")
    PURPLE = ColorFormat("purple", "\033[95m")
    RED = ColorFormat("red", "\033[91m")
    VIOLET = ColorFormat("violet", "\033[38;5;177m")
    WHITE = ColorFormat("white", "\033[97m")
    YELLOW = ColorFormat("yellow", "\033[93m")
    RESET = ColorFormat("reset", "\033[0m")

    @classmethod
    def get_ansi_by_name(cls, color_name: str) -> str:
        for attr in dir(cls):
            value = getattr(cls, attr)
            if isinstance(value, ColorFormat):
                if value.name == color_name.lower():
                    return value.ansi
        return cls.RESET.ansi

    @classmethod
    def rainbow(cls, text: str) -> str:
        palette = [
            "\033[91m", "\033[38;5;208m", "\033[93m",
            "\033[92m", "\033[94m", "\033[38;5;177m"
        ]
        result = "".join(
            f"{palette[i % len(palette)]}{char}"
            for i, char in enumerate(text)
        )
        return f"{result}{cls.RESET.ansi}"

    @classmethod
    def bold(cls, text: str) -> str:
        bold: str = '\033[1m'
        reset: str = '\033[0m'
        return f'{bold}{text}{reset}'

    @classmethod
    def apply(cls, style: str | None = None, color_name: str = 'reset',
              text: str = '') -> str:
        if color_name.lower() == "rainbow":
            result = cls.rainbow(text)
        else:
            ansi = cls.get_ansi_by_name(color_name)
            result = f"{ansi}{text}{cls.RESET.ansi}"

        if style == 'bold':
            result = cls.bold(result)

        return result


# ----------------------------------------------------------------------------
#  Specific messages formatting
# ----------------------------------------------------------------------------

def error(text: str) -> None:
    """Format text for error messages."""
    print(Formatter.apply('bold', 'magenta', f'\n>>> ERROR! {text}'))


def warning(text: str) -> None:
    """Format text for warning messages."""
    print(Formatter.apply('bold', 'yellow', f'\n>>> WARNING! {text}'))
