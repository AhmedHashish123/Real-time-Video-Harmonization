"""
An implementaion for the paper "Color Harmonization for Videos"
http://graphics.stanford.edu/~niloy/research/vid_harmonize/videoHarmonize_icvgip_08.html

Some details were filled in from the paper "Color Harmonization"
https://www.microsoft.com/en-us/research/wp-content/uploads/2006/07/harmonization.pdf
"""

import torch
from math import floor, pi

MAX = 2 * pi    # Hue values fall in the range [0, MAX]
BINS = 100      # Number of levels for hue values range discretization
E = 1e-06       # Small constant to ensure numerical stability

def circular_match(data, lower, upper, start, end, modulus):
    """Linearly transforms circular data (within modulus) in range [start, end] to the range [lower, upper]"""
    mask = (data > start) & (data < end) if start < end else (data > start) | (data < end)
    data[mask] = (((upper - lower) % modulus) / ((end - start) % modulus) * ((data[mask] - start) % modulus) + lower) % modulus


class Palette:
    """
    A color palette within the hue space
    A palette is characterized by its type and orientation (can include multiple regions)
    """
    
    class Region:
        """
        A uniform region within the hue space
        A region is characterized by its center and width (percentage of the space)
        """
        
        def __init__(self, center, percentage):
            """A simple constructor"""
            self.center = center
            self.percentage = percentage
            self.half_arc = percentage / 2 * MAX
        
        def potential(self, hsv):
            """Calculates how much of the given image fits the region"""
            return (((((hsv[0, ...] - self.center) % MAX) <= self.half_arc) |
                     (((self.center - hsv[0, ...]) % MAX) <= self.half_arc)) * hsv[1, ...]).sum()

        def apply(self, hsv, start, end):
            """Matches the given image colors (in range [start, end]) to the region"""
            circular_match(
                hsv[0, ...],
                (self.center - self.half_arc) % MAX,
                (self.center + self.half_arc) % MAX,
                start,
                end,
                MAX
            )

    @staticmethod
    def range(start, end):
        """Returns a discrete iterator over the range [start, end]"""
        i = 0
        while i < (end - start) % MAX:
            yield (start + i) % MAX
            i += MAX / BINS

    @staticmethod
    def optimize_s(s, hist, percentage):
        """Moves the separation point to the most suitable location in the histogram"""
        delta = percentage * MAX / 8  # Empirical value
        return min(
            (s_ for s_ in Palette.range((s - delta) % MAX, (s + delta) % MAX)),
            key=lambda s: hist[floor(s / MAX * BINS)]
        )
    
    @staticmethod
    def reflect(value, ref=None):
        """Reflects a point within a circular range with reference to another point; if no reference is given, the point is rotated half a circle"""
        if ref is None:
            ref = MAX / 4 + value  # makes half a circle rotation (value + MAX / 2)
        return (value + 2 * (ref - value)) % MAX

    @classmethod
    def extract(cls, hsv, palette_types=None):
        """Constructs a color palette by choosing the best type and orientation matching an image"""
        if palette_types is None: palette_types = ('I', 'Y', 'L')  # Sensible selection of palette types
        return max(
            (cls(t, theta) for t in palette_types for theta in Palette.range(0, MAX - E)),
            key=lambda palette: palette.potential(hsv)
        )                

    def __init__(self, t, theta):
        """A constructor with initializations and calculations needed later"""

        self.regions = {
            # Empirical values
            'i': (Palette.Region(theta, 0.15),),
            'V': (Palette.Region(theta, 0.30),),
            'T': (Palette.Region(theta, 0.50),),
            'I': (Palette.Region(theta, 0.15), Palette.Region(Palette.reflect(theta), 0.15)),
            'X': (Palette.Region(theta, 0.25), Palette.Region(Palette.reflect(theta), 0.25)),
            'Y': (Palette.Region(theta, 0.30), Palette.Region(Palette.reflect(theta), 0.15)),
            'L': (Palette.Region(theta, 0.15), Palette.Region(theta + MAX / 4, 0.30))  # % MAX is removed on purpose for the s computation to be consistent
        }[t]
        
        if len(self.regions) == 1:
            self.s = Palette.reflect(self.regions[0].center)
        else:
            region1, region2 = self.regions[0], self.regions[1]
            self.avg_percentage = (region1.percentage + region2.percentage) / 2
            self.s1 = (region1.center * region2.percentage + region2.center * region1.percentage) / (2 *  self.avg_percentage)
            if t == 'L':
                self.s2 = Palette.reflect(self.s1)
            else:
                self.s2 = Palette.reflect(self.s1, ref=max(region1.center, region2.center))
                if theta >= MAX / 2: self.s1, self.s2 = self.s2, self.s1

    def potential(self, hsv):
        """Calculates how much of the given image fits the palette"""
        return sum(region.potential(hsv) for region in self.regions) / (sum(region.percentage for region in self.regions) + 10.0)  # TODO better potential

    def apply(self, hsv):
        """Matches the given image colors to the palette"""
        hist = torch.histc(hsv[0, ...], BINS, 0, MAX)
        if len(self.regions) == 1:
            s = Palette.optimize_s(self.s, hist, self.regions[0].percentage)
            self.regions[0].apply(hsv, s, (s - E) % MAX)
        else:
            s1 = Palette.optimize_s(self.s1, hist, self.avg_percentage)
            s2 = Palette.optimize_s(self.s2, hist, self.avg_percentage)
            self.regions[0].apply(hsv, s2, s1)
            self.regions[1].apply(hsv, s1, s2)
