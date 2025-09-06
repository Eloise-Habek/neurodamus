"""
Stimuli sources. inc current and conductance sources which can be attached to cells
"""

from __future__ import absolute_import
from . import Neuron
from .random import RNG, gamma
import logging
import h5py
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from neuron import h
from scipy.spatial.transform import Rotation as R


class SignalSource:

    def __init__(self, base_amp=0.0, *, delay=0, rng=None):
        """
        Creates a new signal source, which can create composed signals
        Args:
            base_amp: The base (resting) amplitude of the signal (Default: 0)
            rng: The Random Number Generator. Used in the Noise functions
        """
        h = Neuron.h
        self.stim_vec = h.Vector()
        self.stim_vec2 = h.Vector()
        self.time_vec = h.Vector()
        self._cur_t = 0
        self._base_amp = base_amp
        self._rng = rng
        if delay > .0:
            self._add_point(base_amp)
            self._cur_t = delay

    def reset(self):
        self.stim_vec.resize(0)
        self.time_vec.resize(0)

    def _add_point(self, amp):
        """Appends a single point to the time-signal source.
        Note: It doesnt advance time, not supposed to be called directly
        """
        self.time_vec.append(self._cur_t)
        self.stim_vec.append(amp)

    def delay(self, duration):
        """Increments the ref time so that the next created signal is delayed
        """
        # NOTE: We rely on the fact that Neuron allows "instantaneous" changes
        # and made all signal shapes return to base_amp. Therefore delay() doesn't
        # need to introduce any point to avoid interpolation.
        self._cur_t += duration
        return self

    def add_segment(self, amp, duration, amp2=None):
        """Sets a linear signal for a certain duration.

        If amp2 is None (default) then we have constant signal
        """
        self._add_point(amp)
        self.delay(duration)
        self._add_point(amp if amp2 is None else amp2)
        return self

    def add_pulse(self, max_amp, duration, **kw):
        """Adds a pulse.

        A pulse is characterized by raising from a base amplitude, for a certain duration.
        """
        base_amp = kw.get("base_amp", self._base_amp)
        self._add_point(base_amp)
        self.add_segment(max_amp, duration)
        self._add_point(base_amp)
        return self

    def add_ramp(self, amp1, amp2, duration, **kw):
        """Adds a ramp.

        A ramp is characterized by a pulse whose peak changes uniformly during its length.
        Neuron automatically interpolates all values between [t0, t1] as a ramp
        """
        base_amp = kw.get("base_amp", self._base_amp)

        self._add_point(base_amp)

        delay = kw.get("delay",0)
        self.delay(delay)

        self.add_segment(amp1, duration, amp2)
        self._add_point(base_amp)
        return self

    def add_train(self, amp, frequency, pulse_duration, total_duration, **kw):
        """Stimulus with repeated pulse injections at a specified frequency.

        Args:
            amp: the amplitude of a each pulse
            frequency: determines the number of pulses per second (hz)
            pulse_duration: the duration of a single pulse (peak time) (ms)
            total_duration: duration of the whole train (ms)
            base_amp: The base amplitude
        """
        base_amp = kw.get("base_amp", self._base_amp)

        tau = 1000 / frequency
        delay = tau - pulse_duration
        number_pulses = int(total_duration / tau)
        for _ in range(number_pulses):
            self.add_pulse(amp, pulse_duration, base_amp=base_amp)
            self.delay(delay)

        # Add final pulse, possibly partial
        remaining_time = total_duration - number_pulses * tau
        if pulse_duration <= remaining_time:
            self.add_pulse(amp, pulse_duration, base_amp=base_amp)
            self.delay(min(delay, remaining_time - pulse_duration))
        else:
            self.add_pulse(amp, remaining_time, base_amp=base_amp)
        # Last point
        self._add_point(base_amp)
        return self


    def add_sin(self, amp, total_duration, freq, step=0.025, **kw):
        """ Builds a sinusoidal signal.
        Args:
            amp: The max amplitude of the wave
            total_duration: Total duration, in ms
            freq: The wave frequency, in Hz
            step: The step, in ms (default: 0.025)
        """


        base_amp = kw.get("base_amp", self._base_amp)
        delay = kw.get("delay",0)
        self.delay(delay)

        tvec = Neuron.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + total_duration, step)
        self.time_vec.append(tvec)
        self.delay(total_duration)

        stim = Neuron.h.Vector(len(tvec))


        stim.sin(freq, 0.0, step)
        stim.mul(amp[0])

        self.stim_vec.append(stim)

        self._add_point(base_amp)  # Last point

        return self

    def add_sines(self, total_duration, freq, freq1=0, step=0.025, **kw):
        """ Builds a sinusoidal signal from a combination of sines.
        Args:
            total_duration: Total duration, in ms, including ramp-up and ramp-down periods
            freq: The wave frequency, in Hz
            step: The step, in ms (default: 0.025)
        """

        base_amp = kw.get("base_amp", self._base_amp)
        delay = kw.get("delay",0)
        self.delay(delay)

        tvec = Neuron.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + total_duration, step)
        self.time_vec.append(tvec)
        self.delay(total_duration)

        stim = Neuron.h.Vector(len(tvec))

        stim.sin(freq, 0.0, step)

        self.stim_vec.append(stim)
        self._add_point(base_amp)  # Last point

        stim1 = Neuron.h.Vector(len(tvec))

        stim1.sin(freq, 0.0, step)

        self.stim_vec2.append(stim1)
        self.stim_vec2.append(base_amp)

        return self

    def add_sinspec(self, start, dur):
        raise NotImplementedError()

    def add_pulses(self, pulse_duration, amp, *more_amps, **kw):
        """Appends a set of pulsed signals without returning to zero
           Each pulse is applied 'dur' time.

        Args:
          pulse_duration: The duration of each pulse
          amp: The amplitude of the first pulse
          *more_amps: 2nd, 3rd, ... pulse amplitudes
          **kw: Additional params:
            - base_amp [default: 0]
        """
        # First and last are base_amp
        base_amp = kw.get("base_amp", self._base_amp)
        self._add_point(base_amp)
        self.add_segment(amp, pulse_duration)
        for amp in more_amps:
            self.add_segment(amp, pulse_duration)
        self._add_point(base_amp)
        return self

    def add_noise(self, mean, variance, duration, dt=0.5):
        """Adds a noise component to the signal.
        """
        rng = self._rng or RNG()  # Creates a default RNG
        if not self._rng:
            logging.warning("Using a default RNG for noise generation")
        rng.normal(mean, variance)
        tvec = Neuron.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + duration, dt)
        svec = Neuron.h.Vector(len(tvec))
        svec.setrand(rng)

        # Delimit noise signals with base_amp
        # Otherwise Neuron does interpolation with surrounding points
        self._add_point(self._base_amp)
        self.time_vec.append(tvec)
        self.stim_vec.append(svec)
        self._cur_t += duration
        self._add_point(self._base_amp)
        return self

    def add_shot_noise(self, tau_D, tau_R, rate, amp_mean, amp_var, duration, dt=0.25):
        """
        Adds a Poisson shot noise signal with gamma-distributed amplitudes and
        bi-exponential impulse response.

        tau_D: bi-exponential decay time [ms]
        tau_R: bi-exponential rise time [ms]
        rate: Poisson event rate [Hz]
        amp_mean: mean of gamma-distributed amplitudes [nA]
        amp_var: variance of gamma-distributed amplitudes [nA^2]
        duration: duration of signal [ms]
        dt: timestep [ms]
        """
        from math import sqrt, exp, log

        rng = self._rng or RNG()  # Creates a default RNG
        if not self._rng:
            logging.warning("Using a default RNG for shot noise generation")

        tvec = Neuron.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + duration, dt)  # time vector
        ntstep = len(tvec)  # total number of timesteps

        rate_ms = rate / 1000  # rate in 1 / ms [mHz]
        napprox = 1 + int(duration * rate_ms)       # approximate number of events, at least one
        napprox = int(napprox + 3 * sqrt(napprox))  # better bound, as in elephant

        exp_scale = 1 / rate  # scale parameter of exponential distribution of time intervals
        rng.negexp(exp_scale)
        iei = Neuron.h.Vector(napprox)
        iei.setrand(rng)  # generate inter-event intervals

        ev = Neuron.h.Vector()
        ev.integral(iei, 1).mul(1000)  # generate events in ms
        # add events if last event falls short of duration
        while ev[-1] < duration:
            iei_new = Neuron.h.Vector(100)  # generate 100 new inter-event intervals
            iei_new.setrand(rng)            # here rng is still negexp
            ev_new = Neuron.h.Vector()
            ev_new.integral(iei_new, 1).mul(1000).add(ev[-1])  # generate new shifted events in ms
            ev.append(ev_new)  # append new events
        ev.where("<", duration)  # remove events exceeding duration
        ev.div(dt)  # divide events by timestep

        nev = Neuron.h.Vector([round(x) for x in ev])  # round to integer timestep index
        nev.where("<", ntstep)  # remove events exceeding number of timesteps

        sign = 1
        # if amplitude mean is negative, invert sign of current
        if amp_mean < 0:
            amp_mean = -amp_mean
            sign = -1

        gamma_scale = amp_var / amp_mean      # scale parameter of gamma distribution
        gamma_shape = amp_mean / gamma_scale  # shape parameter of gamma distribution
        # sample gamma-distributed amplitudes
        amp = gamma(rng, gamma_shape, gamma_scale, len(nev))

        E = Neuron.h.Vector(ntstep, 0)  # full signal
        for n, A in zip(nev, amp):
            E.x[int(n)] += sign * A  # add impulses, may overlap due to rounding to timestep

        # perform equivalent of convolution with bi-exponential impulse response
        # through a composite autoregressive process with impulse train as innovations

        # unitless quantities (time measured in timesteps)
        a = exp(-dt / tau_D)
        b = exp(-dt / tau_R)
        D = -log(a)
        R = -log(b)
        t_peak = log(R / D) / (R - D)
        A = (a / b - 1) / (a ** t_peak - b ** t_peak)

        P = Neuron.h.Vector(ntstep, 0)
        B = Neuron.h.Vector(ntstep, 0)

        # composite autoregressive process with exact solution
        # P[n] = b * (a ^ n - b ^ n) / (a - b)
        # for unit response B[0] = P[0] = 0, E[0] = 1
        for n in range(1, ntstep):
            P.x[n] = a * P[n - 1] + b * B[n - 1]
            B.x[n] = b * B[n - 1] + E[n - 1]

        P.mul(A)  # normalize to peak amplitude

        self._add_point(self._base_amp)
        self.time_vec.append(tvec)
        self.stim_vec.append(P)
        self._cur_t += duration
        self._add_point(self._base_amp)

        return self

    def add_ornstein_uhlenbeck(self, tau, sigma, mean, duration, dt=0.25):
        """
        Adds an Ornstein-Uhlenbeck process with given correlation time,
        standard deviation and mean value.

        tau: correlation time [ms], white noise if zero
        sigma: standard deviation [uS]
        mean: mean value [uS]
        duration: duration of signal [ms]
        dt: timestep [ms]
        """
        from math import sqrt, exp

        rng = self._rng or RNG()  # Creates a default RNG
        if not self._rng:
            logging.warning("Using a default RNG for Ornstein-Uhlenbeck process")

        tvec = Neuron.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + duration, dt)  # time vector
        ntstep = len(tvec)  # total number of timesteps

        svec = Neuron.h.Vector(ntstep, 0)  # stim vector

        noise = Neuron.h.Vector(ntstep)  # Gaussian noise
        rng.normal(0.0, 1.0)
        noise.setrand(rng)  # generate Gaussian noise

        if tau < 1e-9:
            svec = noise.mul(sigma)  # white noise
        else:
            mu = exp(-dt / tau)  # auxiliar factor [unitless]
            A = sigma * sqrt(1 - mu * mu)  # amplitude [uS]
            noise.mul(A)  # scale noise by amplitude [uS]

            # Exact update formula (independent of dt) from Gillespie 1996
            for n in range(1, ntstep):
                svec.x[n] = svec[n - 1] * mu + noise[n]  # signal [uS]

        svec.add(mean)  # shift signal by mean value [uS]

        self._add_point(self._base_amp)
        self.time_vec.append(tvec)
        self.stim_vec.append(svec)
        self._cur_t += duration
        self._add_point(self._base_amp)

        return self

    # PLOTTING
    def plot(self, ylims=None):
        from matplotlib import pyplot
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)  # (nrows, ncols, axnum)
        ax.plot(self.time_vec, self.stim_vec, label="Signal amplitude")
        ax.legend()
        if ylims:
            ax.set_ylim(*ylims)
        fig.show()

    # ==== Helpers =====
    # Helper methods forward generic kwargs to base class, like rng and delay

    @classmethod
    def pulse(cls, max_amp, duration, base_amp=.0, **kw):
        return cls(base_amp, **kw).add_pulse(max_amp, duration)

    @classmethod
    def ramp(cls, amp1, amp2, duration, base_amp=.0, **kw):
        return cls(base_amp, **kw).add_ramp(amp1, amp2, duration)

    @classmethod
    def train(cls, amp, frequency, pulse_duration, total_duration, base_amp=.0, **kw):
        return cls(base_amp, **kw).add_train(amp, frequency, pulse_duration, total_duration)

    @classmethod
    def sin(cls, amp, total_duration, freq, step=0.025, base_amp=.0, **kw):
        return cls(base_amp, **kw).add_sin(amp, total_duration, freq, step)

    @classmethod
    def noise(cls, mean, variance, duration, dt=0.5, base_amp=.0, **kw):
        return cls(base_amp, **kw).add_noise(mean, variance, duration, dt)

    @classmethod
    def shot_noise(cls, tau_D, tau_R, rate, amp_mean, var, duration, dt=0.25, base_amp=.0, **kw):
        return cls(base_amp, **kw).add_shot_noise(tau_D, tau_R, rate, amp_mean, var, duration, dt)

    @classmethod
    def ornstein_uhlenbeck(cls, tau, sigma, mean, duration, dt=0.25, base_amp=.0, **kw):
        return cls(base_amp, **kw).add_ornstein_uhlenbeck(tau, sigma, mean,duration, dt)

    # Operations
    def __add__(self, other):
        """# Adds signals. Two added signals sum amplitudes"""
        raise NotImplementedError("Adding signals is not available yet")


class CurrentSource(SignalSource):
    _all_sources = []

    def __init__(self, base_amp=0.0, *, delay=0, rng=None):
        """
        Creates a new current source that injects a signal under IClamp
        """
        super().__init__(base_amp, delay=delay, rng=rng)
        self._clamps = set()
        self._all_sources.append(self)

    class _Clamp:
        def __init__(self, cell_section, position=0.5, clamp_container=None,
                     stim_vec_mode=True, time_vec=None, stim_vec=None,
                     **clamp_params):
            self.clamp = Neuron.h.IClamp(position, sec=cell_section)
            if stim_vec_mode:
                assert time_vec is not None and stim_vec is not None
                self.clamp.dur = time_vec[-1]
                stim_vec.play(self.clamp._ref_amp, time_vec, 1)
            else:
                for param, val in clamp_params.items():
                    setattr(self.clamp, param, val)
            # Clamps must be kept otherwise they are garbage-collected
            self._all_clamps = clamp_container
            clamp_container.add(self)

        def detach(self):
            """Detaches a clamp from a cell, destroying it"""
            self._all_clamps.discard(self)
            del self.clamp  # Force del on the clamp (there might be references to self)

    def attach_to(self, section, position=0.5):
        return CurrentSource._Clamp(section, position, self._clamps, True,
                                    self.time_vec, self.stim_vec)

    # Constant has a special attach_to and doesnt share any composing method
    class Constant:
        """Class implementing a minimal IClamp for a Constant current."""
        _clamps = set()

        def __init__(self, amp, duration, delay=0):
            self._amp = amp
            self._dur = duration
            self._delay = delay

        def attach_to(self, section, position=0.5):
            return CurrentSource._Clamp(section, position, self._clamps, False,
                                        amp=self._amp, delay=self._delay, dur=self._dur)


class ConductanceSource(SignalSource):
    _all_sources = []

    def __init__(self, reversal=0.0, *, delay=.0, rng=None):
        """
        Creates a new conductance source that injects a conductance by driving
        the rs of an SEClamp at a given reversal potential.

        reversal: reversal potential of conductance (mV)
        """
        super().__init__(0.0, delay=delay, rng=rng)  # set SignalSource's base_amp to zero
        self._reversal = reversal   # set reversal from base_amp parameter in classmethods
        self._clamps = set()
        self._all_sources.append(self)

    class _DynamicClamp:
        def __init__(self, cell_section, position=0.5, clamp_container=None,
                     stim_vec_mode=True, time_vec=None, stim_vec=None,
                     reversal=0.0, **clamp_params):
            self.clamp = Neuron.h.SEClamp(position, sec=cell_section)
            if stim_vec_mode:
                assert time_vec is not None and stim_vec is not None
                self.clamp.dur1 = time_vec[-1]
                self.clamp.amp1 = reversal
                # support delay with initial zero
                self.time_vec = Neuron.h.Vector(1, 0).append(time_vec)
                self.stim_vec = Neuron.h.Vector(1, 0).append(stim_vec)
                # replace self.stim_vec with inverted and clamped signal
                # rs is in MOhm, so conductance is in uS (micro Siemens)
                self.stim_vec = Neuron.h.Vector(
                    [1 / x if x > 1E-9 and x < 1E9 else 1E9 for x in self.stim_vec])
                self.stim_vec.play(self.clamp._ref_rs, self.time_vec, 1)
            else:
                for param, val in clamp_params.items():
                    setattr(self.clamp, param, val)
            # Clamps must be kept otherwise they are garbage-collected
            self._all_clamps = clamp_container
            clamp_container.add(self)

        def detach(self):
            """Detaches a clamp from a cell, destroying it"""
            self._all_clamps.discard(self)
            del self.clamp  # Force del on the clamp (there might be references to self)

    def attach_to(self, section, position=0.5):
        return ConductanceSource._DynamicClamp(section, position, self._clamps, True,
                                               self.time_vec, self.stim_vec, self._reversal)


# EStim class is a derivative of TStim for stimuli with an extracelular electrode. The main
# difference is that it collects all elementary stimuli pulses and converts them using a
# VirtualElectrode object before it injects anything
#
# The stimulus is defined on the hoc level by using the addpoint function for every (step) change
# in extracellular electrode voltage. At this stage only step changes can be used. Gradual,
# i.e. sinusoidal changes will be implemented in the future
# After every step has been defined, you have to call initElec() to perform the frequency dependent
# transformation. This transformation turns e_electrode into e_extracellular at distance d=1 micron
# from the electrode. After the transformation is complete, NO MORE STEPS CAN BE ADDED!
# You can then use inject() to first scale e_extracellular down by a distance dependent factor
# and then vector.play() it into the currently accessed compartment
#
# TODO: 1. more stimulus primitives than step. 2. a dt of 0.1 ms is hardcoded. make this flexible!

class ElectrodeSource(SignalSource):
    _all_sources = []

    def __init__(self, delay, duration, Ex_0, Ey_0, Ez_0, frequency0,
                 Ex_1, Ey_1, Ez_1, frequency1,
                 ramp_up_time, ramp_down_time):


        """
        Creates a new source that injects a signal under e_extracellular
        """
        super().__init__()
        self.stim_delay = delay
        self.duration = duration

        self.Ex_0 = Ex_0 # x-component of the first E field (in V/m)
        self.Ey_0 = Ey_0 # y-component of the first E field (in V/m)
        self.Ez_0 = Ez_0 # z-component of the first E field (in V/m)
        self.frequency0 = frequency0 # Temporal frequency of the first E field (in Hz)
        self.Ex_1 = Ex_1 # x-component of the second E field (in V/m)
        self.Ey_1 = Ey_1 # y-component of the second E field (in V/m)
        self.Ez_1 = Ez_1 # z-component of the second E field (in V/m)
        self.frequency1 = frequency1 # Temporal frequency of the second E field (in Hz)

        self._all_sources.append(self)
        self.extracellulars = []

        self.ramp_up_time = ramp_up_time # Time over which the stimulus ramps up to its maximum amplitude (in ms)
        self.ramp_down_time = ramp_down_time # Time over which the stimulus ramps down to zero (in ms)

        self.axon1 = False #  # Indicates whether the E field has already been interpolated for the first axonal segment

        self.add_sines( self.duration+self.ramp_up_time+self.ramp_down_time, self.frequency0,self.frequency1,delay=self.stim_delay, step=self.stepSize) # Defines the temporal profile of the signal


    def get_soma_position(self,section):

        '''
        If the given segment is a soma, then we calculate its position by averaging all of the 3d points associated with it
        '''

        n3d = section.n3d()
        xpos = []
        ypos = []
        zpos = []

        for n in range(n3d):
            xpos.append(section.x3d(n))
            ypos.append(section.y3d(n))
            zpos.append(section.z3d(n))

        x = np.mean(xpos)
        y = np.mean(ypos)
        z = np.mean(zpos)

        print(np.array([x,y,z]),flush=True)

        return np.array([x,y,z])

    def grindaway(self,hsection):
        """Grindaway"""

        # get the data for the section
        n_segments = int(h.n3d(sec=hsection))
        n_comps = hsection.nseg

        xs = np.zeros(n_segments)
        ys = np.zeros(n_segments)
        zs = np.zeros(n_segments)
        lengths = np.zeros(n_segments)
        for index in range(0, n_segments):
            xs[index] = h.x3d(index, sec=hsection)
            ys[index] = h.y3d(index, sec=hsection)
            zs[index] = h.z3d(index, sec=hsection)
            lengths[index] = h.arc3d(index, sec=hsection)

        # to use Vector class's .interpolate()
        # must first scale the independent variable
        # i.e. normalize length along centroid
        lengths /= (lengths[-1])

        # initialize the destination "independent" vector
        # range = np.array(n_comps+2)
        comp_range = np.arange(0, n_comps + 2) / n_comps - \
            1.0 / (2 * n_comps)
        comp_range[0] = 0
        comp_range[-1] = 1

        # length contains the normalized distances of the pt3d points
        # along the centroid of the section.  These are spaced at
        # irregular intervals.
        # range contains the normalized distances of the nodes along the
        # centroid of the section.  These are spaced at regular intervals.
        # Ready to interpolate.

        xs_interp = np.interp(comp_range, lengths, xs)
        ys_interp = np.interp(comp_range, lengths, ys)
        zs_interp = np.interp(comp_range, lengths, zs)

        return xs_interp, ys_interp, zs_interp

    def get_positions(self,
            hsection1=None,
            location1=None):
        """Gets position on a section

        Parameters
        ----------

        hsection1 : hoc section
                    First section
        location1 : float
                    range x along hsection1
        """

        xs_interp1, ys_interp1, zs_interp1 = self.grindaway(hsection1)

        x1 = xs_interp1[int(np.floor((len(xs_interp1) - 1) * location1))]
        y1 = ys_interp1[int(np.floor((len(ys_interp1) - 1) * location1))]
        z1 = zs_interp1[int(np.floor((len(zs_interp1) - 1) * location1))]

        pos1 = np.array([x1,y1,z1])

        return pos1


    def interp_axon_positions(self,section,x):

        '''
        If the given section is an axon, then we need to guess where it is located
        '''

        # Specifies list of points for each Cartesian coordinate
        xpos = []
        ypos = []
        zpos = []
        lens = []

        ### Adds soma position to the list of coordinates
        xpos.append(self.soma_position[0])
        ypos.append(self.soma_position[1])
        zpos.append(self.soma_position[2])
        lens.append(0)

        # We assume that the axon is oriented along the z-axis, so we maintain the x- and y-coordinates of the soma
        xpos.append(self.soma_position[0])
        ypos.append(self.soma_position[1])
        lens.append(1)

        if self.axon1 == False:
            zpos.append(self.soma_position[2]+30) # If this is the first axonal segment, then it is 30 um displaced along the z-axis
            self.axon1 = True
        else:
            zpos.append(self.soma_position[2]+60) # If it is the second axonal segment, then it is displaced by 60 um

        # Then, we interpolate the coordinates for the given location x along the segment
        fX = interp1d(lens,xpos)
        segX = fX(x)
        fY = interp1d(lens,ypos)
        segY = fY(x)
        fZ = interp1d(lens,zpos)
        segZ = fZ(x)

        segpos = np.array([segX,segY,segZ])


        return segpos


    def apply_ramp(self, vector, step=0.025):

        ramp_up_number = int(self.ramp_up_time/step) # Number of time points during the ramp-up window
        ramp_down_number = int(self.ramp_down_number/step) # Number of time points during the ramp-down window

        if ramp_up_number > 0:
            ramp_up = np.linspace(0, 1, ramp_up_number)
            vector[:ramp_up_number] *= ramp_up
        if ramp_down_number > 0:
            ramp_down = np.linspace(1, 0, ramp_down_number)
            vector[len(vector) - ramp_down_number:] *= ramp_down

        return vector

    def get_scale_factor(self, section, x):

        if 'soma' in section.name():

            segpositions = self.get_soma_position(section)

            self.soma_position = segpositions.copy()
        else:

            if int(h.n3d(sec=section)) == 0: # Axonal segments don't have 3d points associated, so we guess
                segpositions = self.interp_axon_positions(section, x)
            else:
                segpositions = self.get_positions(section, x)

        if isinstance(self.offset, np.ndarray):
            segpositions += self.offset * 1e3  # offset in mm converted to um

        self.new_soma_pos = self.soma_position.copy()

        scaleFactor0, scaleFactor1 = self.uniform_potentials(segpositions)

        return scaleFactor0, scaleFactor1

    def uniform_potentials(self, segpositions):

        # Calculates distance between soma and each segment, since the ground is assumed to be at the soma

        displacementVector = segpositions - self.soma_position
        displacementVector *= 1e-6 # Converts from um to m

        scaleFactor0 = np.dot(displacementVector, np.array([self.Ex_0,self.Ey_0, self.Ez_0]))
        scaleFactor0 *= 1e3 # Converts from V to mV

        scaleFactor1 = np.dot(displacementVector, np.array([self.Ex_1, self.Ey_1, self.Ez_1]))
        scaleFactor1 *= 1e3  # Converts from V to mV

        return scaleFactor0, scaleFactor1

    def attach_to(self, section, x):

        self.extracellulars.append(self.time_vec)

        section.insert('extracellular')

        scaleFac0, scaleFac1 = self.get_scale_factor(section, x) # Calculates the potential relative to the soma for the given segment, for both of the E fields

        stimVec0 = self.stim_vec.to_python()
        stimVec0 = self.apply_ramp(stimVec0) # Scales the sinusoid by the ramp-up and ramp-down windows
        stimVec0 *= scaleFac0 # Applies the calculated potential to the temporal waveform

        stimVec1 = self.stim_vec.to_python()
        stimVec1 = self.apply_ramp(stimVec1)# Scales the sinusoid by the ramp-up and ramp-down windows
        stimVec1 *= scaleFac1# Applies the calculated potential to the temporal waveform

        stimVec = stimVec0 + stimVec1 # The total signal is just the sum of the contribution from the two E fields

        segVec = h.Vector()

        for v in stimVec:
            segVec.append(v)


        self.extracellulars.append(segVec)
        self.extracellulars.append(seg.extracellular)
        self.extracellulars.append(seg.extracellular.e)

        out = segVec.play(seg.extracellular._ref_e, self.time_vec)
        self.extracellulars.append(out)

        return segVec.to_python(), self.time_vec.to_python(), newpos



