import numpy as np
import h5py

class NbodyLoader:
    def __init__(self, path):
        self.path = path

    def load_nbody(self):
        with h5py.File(self.path, 'r') as file:
            part = file['PartType1']
            pos = part['Coordinates'][()]
            vel = part['Velocities'][()]
            pid = part['ParticleIDs'][()] - 1

            header = file['Header']
            box_size = header.attrs['BoxSize']
            redshift = header.attrs['Redshift']
            num_particles = header.attrs['NumPart_Total'][1]
            Mass = header.attrs['MassTable'][1]

        sort = np.argsort(pid)
        return {
            'pos': pos[sort],
            'vel': vel[sort],
            'id': pid[sort],
            'box_size': box_size,
            'redshift': redshift,
            'num_particles': num_particles,
            'mass': Mass
        }
