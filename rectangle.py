class Rectangle:
    
    __slots__ = '_xmin', '_ymin', '_xmax', '_ymax', '_width', '_height'

    def __init__(self, xmin, ymin, width, height):
        self._xmin = xmin
        self._ymin = ymin
        self._width = width
        self._height = height
        self._xmax = self._xmin + self.width
        self._ymax = self._ymin + self.height

    @property
    def xmin(self):
        return self._xmin
    
    @property
    def ymin(self):
        return self._ymin
    

    @property
    def xmax(self):
        return self._xmax

    @property
    def ymax(self):
        return self._ymax
    
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height

    def intersects(self, x, y):
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def __repr__(self):
        return f"{(self.xmin, self.ymin)} {(self.xmax, self.ymax)}"
