import math

class SensorArray:
    """
     A fan of raycast sensors simulating a forward-facing camera's field of view.
    """

    def __init__(self, count=5, fov_degrees=120, max_distance=200, step=4):
        self.max_distance = max_distance
        self.step = step

        #Calculate each sensors offset
        if count == 1:
            self.angle_offsets = [0.0]
        else:
            half_fov = fov_degrees/2
            self.angle_offsets = [-half_fov + i * (fov_degrees/(count - 1)) for i in range(count)]

    def sense(self, origin_x, origin_y, heading_degrees, track):
        """Casts all rays from origin x, y facing to the heading degrees. 
        Returns a list of distances in the order of offsets"""
        return [self._cast_ray(origin_x, origin_y, heading_degrees + offset, track)
                for offset in self.angle_offsets]

    def _cast_ray(self, x, y, angle_degrees, track):
        """The private function that checks the distance"""
        rad = math.radians(angle_degrees)
        dx, dy = math.cos(rad), math.sin(rad)

        distance = 0
        while distance <= self.max_distance:
            test_x = x + dx * distance
            test_y = y + dy * distance
            if not track.is_on_track(test_x, test_y):
                return float(distance/self.max_distance)
            distance += self.step

        return 1.0

    def get_endpoints(self, origin_x, origin_y, heading_degrees, distances):
        """
        Given the distances from sense(), returns the actual (x, y) endpoint
        of each ray - purely for debug drawing, not used by the sensing logic.
        """
        endpoints = []
        for offset, distance in zip(self.angle_offsets, distances):
            rad = math.radians(heading_degrees + offset)
            endpoints.append((
                origin_x + math.cos(rad) * distance,
                origin_y + math.sin(rad) * distance,
            ))
        return endpoints
