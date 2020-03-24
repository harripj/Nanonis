import numpy as _np
import pandas as _pd
from datetime import datetime as _datetime
import string as _string
import re as _re
import os as _os
from copy import deepcopy as _deepcopy
from matplotlib import pyplot as _plt
from scipy import constants as _constants


class SXM:
    """
    Opens SXM files from Nanonis controller.
    
    Image data is stored as numpy array, accessed with SXM.data.
    If more than one channel N then the data is stacked in the first dimension, ie. (N, i, j).
    
    Header is accessed with SXM.header.
    
    """

    def __init__(self, fname):
        """

        Parameters
        ----------
        fname: str
            Path to .sxm file.
        
        """

        # define attributes of class, ie a dictionary, image array to store the image
        # and the channels recorded in the experiment -- fwd, bwd and any lock-ins
        self.fname = fname
        self.header = dict()  # dictionary of all sxm header settings
        self.data = None  # current image parsed

        # parse header
        self.read_header()
        # mae pixels int
        self.header["SCAN_PIXELS"] = [int(i) for i in self.header["SCAN_PIXELS"]]

        # get column for direction data, ie fwd, bwd, or both
        index_direction = [i for i in self.header["DATA_INFO"][0]].index("Direction")
        index_name = [i for i in self.header["DATA_INFO"][0]].index("Name")

        self.channels = []

        for i in range(1, len(self.header["DATA_INFO"])):
            name = self.header["DATA_INFO"][i][index_name]
            both = self.header["DATA_INFO"][i][index_direction] == "both"
            self.channels.extend([name, name] if both else [name])

        #  parse data
        self.read_data()

    def read_header(self):
        """
        
        Opens SXM file, finds header and parses into a dictionary.
        
        """

        values = []

        KEY_FLAG = ":"  # tagged onto start and end of every keyword
        HEADER_END = ":SCANIT_END:"
        DECODE = "UTF-8"

        with open(self.fname, "rb") as f:
            for line in f:
                # remove whitespace and excess colons
                line = line.decode(DECODE).strip()

                # skip empty lines
                if not line:
                    continue

                # start of new keyword line
                if line.startswith(KEY_FLAG):
                    # if values is non-empty then add to dict as new keyword found
                    if values:
                        if len(values) == 1:
                            try:
                                values = float(values[0])
                            except ValueError:
                                values = values[0]
                        else:
                            try:
                                values = [float(x) for x in values]
                            except ValueError:
                                pass
                        if key == "DATA_INFO":
                            self.header[key] = _np.reshape(values, (-1, ncols))
                        else:
                            self.header[key] = values

                    # keyword
                    key = line.strip(KEY_FLAG)
                    # empties the value list as to start a new key value pair
                    values = []

                else:
                    # inline delimiter is whitespace
                    # if > in key then delimiter is actually ;
                    delimiter = ";" if ">" in key else None

                    if key == "DATA_INFO":
                        ncols = len(line.split(delimiter))

                    values.extend(line.split(sep=delimiter))

                # end header file line
                if line == HEADER_END:
                    break

    def read_data(self):
        """
        Opens SXM file, finds data and parses into numpy array.
        
        If more than one channel N then the data is stacked in the first dimension, ie. (N, i, j).
        
        """

        START = b"\x1A\x04"  # indicates start of data
        FORMAT = ">f"  # big-endian float format

        # parse the binary iamge data
        with open(self.fname, "rb") as f:
            data = f.read()  # read all file

            # bytes signify start of images
            # image data starts after start bytes
            start_byte = data.find(START) + len(START)

        self.data = _np.fromfile(self.fname, offset=start_byte, dtype=FORMAT).reshape(
            -1, *self.header["SCAN_PIXELS"]
        )

    def get_time(self):
        """

        Returns acquisition time as datetime.

        """
        return _datetime.strptime(
            "{} {}".format(self.header["REC_DATE"], self.header["REC_TIME"]),
            "%d.%m.%Y %H:%M:%S",
        )

    def get_pixel_size(self):
        """

        Returns pixel size as numpy array, ie. (sz, sy).

        """
        return _np.array(self.header["SCAN_RANGE"]) / self.header["SCAN_PIXELS"]


class STS:
    """

    Opens .dat STS data from Nanonis controller.
    Data is accessed at STS.data as pandas DataFrame.
    Header information is accessed at STS.header.

    """

    @staticmethod
    def broaden_conductance(V, I, dV=0.1):
        """

        Broaden I/V conductance using exponential broadening.

        Parameters
        ----------
        V, I: (N,) ndarray
            Arrays of bias and current values.
        dV: float
            Window width in Volts.

        Returns
        -------
        conductance: (N,) ndarray
            Conductance broadened using smoothed current values.

        References
        ----------
        [1] Feenstra R., Phys. Rev. B, 50 (7), p.4561-4570 (1994)
            DOI: 10.1103/PhysRevB.50.4561
        
        """
        return _np.sum(
            (I / V) * _np.exp(-_np.abs(_np.meshgrid(V, V, indexing="ij")[0] - V) / dV),
            axis=1,
        )

    @staticmethod
    def normalised_differential_conductance(V, I, dIdV, dV=0.1):
        """

        Normalise dI/dV conductance by exponentially broadened I/V.
        Result is (dI/dV) / (I/V)_bar
        Useful for materials with band gaps such as semiconductors.
        Pioneered by Feenstra.

        Parameters
        ----------
        V, I, dIdV: (N,) ndarray
            Arrays of bias, current, and dI/dV values.
        dV: float
            Window width in Volts.

        Returns
        -------
        conductance: (N,) ndarray
            Normalised conductance broadened using smoothed current values.

        References
        ----------
        [1] Feenstra R., Phys. Rev. B, 50 (7), p.4561-4570 (1994)
            DOI: 10.1103/PhysRevB.50.4561
        
        """

        return dIdV / STS.broaden_conductance(V, I, dV)

    def __init__(self, fname, delimiter="\t"):
        """

        Parameters
        ----------
        fname: str
            Path to .dat file.
        delimiter: str, default is '\t'
            Delimiter used for data.

        """

        self.fname = fname
        self._header_length = 0
        self.header = dict()
        self.delimiter = delimiter
        # parse file
        self.read_header()
        self.read_data()

    def read_header(self):
        """
        
        Opens file and reads and parses header information as dictionary.
        
        """

        HEADER_END = "[DATA]"

        header = []

        with open(self.fname) as f:
            for index, line in enumerate(f):
                # remove whitespace and \n
                line = line.strip()
                # increment
                self._header_length = index

                # end of header
                if line == HEADER_END:
                    break

                # get info to test for validity
                temp = line.split(self.delimiter)
                # capitalise key to make consistent between different types of STS
                temp[0] = _string.capwords(temp[0])

                # if no value for key keep in header but make value empty string
                if len(temp) == 1:
                    temp.append("")
                # if there is key and value then include
                if len(temp) == 2:
                    # try to convert value to float if a number
                    try:
                        temp[1] = float(temp[1])
                    # otherwise keep as string
                    except ValueError:
                        pass
                    header.append(temp)

        # create dictionary from parsed header info
        self.header.update(header)

    def read_data(self):
        """
        
        Opens file and reads actual spectra data as pandas DataFrame.
        
        """
        # get spectra data as DataFrame
        # header length + 1 to avoid '[DATA]' line
        self.data = _pd.read_csv(
            self.fname, skiprows=self._header_length + 1, delimiter=self.delimiter
        )

    def get_time(self):
        """

        Returns acquisition time as datetime.

        """
        return _datetime.strptime(self.header["Date"], "%d.%m.%Y %H:%M:%S")

    def plot_spectra(self, ax, channelx=None, channely=None, **kwargs):
        """

        Convenience function to plot STS data on an axes.

        Parameters
        ----------
        ax: plt.Axes
            Axes to plot on.
        channelx, channely: str, int, iterable, or None
            Channel in STS data to plot on x- and y-axis.
            If str must be a column name in STS.data.
            If int then appropriate column is selected.
            channely may be an iterable of length 2.
            If channely is iterable then it must be composed of the other datatypes and they will both be plotted against channelx.
            If None then defaults to search for x-'bias' and y-'current'.
            If they are not found then the first and second columns are used.
            eg. 'LIY 1 omega (A)', 'Bias (V)'.
        **kwargs
            Used for plotting, ie. ax.plot.
            
        """
        # sort out data coulumns
        if channelx is None:
            # search for bias column
            # use regex to find a channel name with 'current'
            # (?i) for case insensitivity
            search = [_re.search("(?i)bias", c) is not None for c in self.data.columns]
            if not any(search):
                # default to second column
                channelx = self.data.columns[0]
            else:
                # get first instance of True result
                channelx = self.data.columns[search.index(True)]
        elif isinstance(channelx, int):
            channelx = self.data.columns[channelx]

        # make list to
        if not isinstance(channely, (list, tuple)):
            channely = [channely]

        # for each channel in list, 2 channels max
        for index, cy in enumerate(channely[:2]):
            if cy is None:
                # find channel to plot
                # use regex to find a channel name with 'current'
                # (?i) for case insensitivity
                search = [
                    _re.search("(?i)current", c) is not None for c in self.data.columns
                ]
                if not any(search):
                    # default to second column
                    cy = self.data.columns[1]
                else:
                    # get first instance of True result
                    cy = self.data.columns[search.index(True)]
            elif isinstance(cy, int):
                cy = self.data.columns[cy]

            # if first channel plot on original axes
            if not index:
                # plot spectra
                ax.plot(
                    self.data[channelx],
                    self.data[cy],
                    label=_os.path.basename(self.fname),
                    **kwargs
                )
                ax.set_ylabel(cy)
            # plot on twinned axes
            else:
                # check for twin
                siblings = ax.get_shared_x_axes().get_siblings(ax)
                if len(siblings) == 1:
                    axt = ax.twinx()
                else:
                    # get axes which is a sibling to ax but is not ax, ie. its twin
                    axt = siblings[[a is not ax for a in siblings].index(True)]
                # plot spectra on twinned x-axes
                axt.plot(
                    self.data[channelx],
                    self.data[cy],
                    label=_os.path.basename(self.fname),
                    color="r",
                )
                axt.set_ylabel(cy)

        ax.set_xlabel(channelx)


def open_file(f):
    """
        
        Returns either SXM or STS object based on filename.
        
        Parameters
        ----------
        f: str
            File name.
            
        Returns
        -------
        out: SXM or STS
            Returns None if file is not .sxm or .dat.
        
        """

    if f.endswith(".sxm"):
        return SXM(f)
    elif f.endswith(".dat"):
        return STS(f)


def organise_spectra(files, plot=False, channelx=None, channely=None, experiment=None):
    """
    
    Normally a folder containing STM images and STS files are fairly messy.
    This function aims to sort that out.
    
    Parameters
    ----------
    files: array-like of str
        Files to sort, can be both .sxm and .dat files mixed.
    channelx, channely: str, int, or None
        Channel in STS data to plot on x- and y-axis.
        If str must be a column name in STS.data.
        If int then appropriate column is selected.
        If None then defaults to search for x-'bias' and y-'current'.
        If they are not found then the first and second columns are used.
        eg. 'LIY 1 omega (A)', 'Bias (V)'.
    experiment: str, array-like of str or None
        If defined then only .dat files including this string 
        in their Experiment value are considered for plotting.
        eg. 'bias spectroscopy' or just 'bias'.
        eg. ['frequency sweep', 'bias spectroscopy']
        
    Returns
    -------
    files: array-like of str
        Files that are sorted by acquisition time.
    
    """

    files = list(sorted(files, key=lambda f: open_file(f).get_time()))

    if plot:
        # sort out experiment kw
        if experiment is not None:
            if isinstance(experiment, str):
                experiment = [experiment.lower()]
            elif isinstance(experiment, (list, tuple)):
                experiment = [exp.lower() for exp in experiment]

        # holder
        spectra = []

        for f in files:
            # open file
            obj = open_file(f)

            # new image in sequence => spectra set to 0
            if isinstance(obj, SXM):
                # found new image
                # if old image had spectra then plot
                if spectra:
                    fig, ax = _plt.subplots(ncols=2, figsize=_plt.figaspect(1.0 / 2))
                    ax[0].matshow(sxm.data[0], cmap=_plt.cm.gray)

                    for s in spectra:
                        # if experiment string is defined, make sure spectroscopy is of right type
                        # skip if string not in STS.header['Experiment']
                        if experiment is not None:
                            if s.header["Experiment"].lower() not in experiment:
                                continue

                        # try to plot on image
                        #  NB. not all .dat files have xy coordinates for some reason...
                        try:
                            # Nanonis uses conventional xy from bottom left, whereas numpy uses ij from top left
                            # => dy * -1
                            delta = (
                                _np.array((s.header["X (m)"], s.header["Y (m)"]))
                                - sxm.header["SCAN_OFFSET"]
                            )
                            delta[1] *= -1

                            # put spectra location on image
                            # SCAN_OFFSET is wrt image center, plot wrt top left
                            p = ax[0].plot(
                                *(
                                    (delta / sxm.get_pixel_size())
                                    + _np.array(sxm.header["SCAN_PIXELS"]) / 2.0
                                ),
                                marker="x",
                                ls="None"
                            )[0]
                            color = p.get_color()
                        except KeyError:
                            # what can you do, hey?
                            color = None
                        # plot spectra on axes
                        s.plot_spectra(
                            ax[1], color=color, channelx=channelx, channely=channely
                        )

                    # set image title as file name
                    ax[0].set_title(_os.path.basename(sxm.fname))
                    ax[1].legend()
                    fig.tight_layout()

                # get new .sxm with no spectra in holder
                sxm = obj
                spectra = []

            # append all spectra acquired from same image into a list
            # spectra associated to current image will be plotted when a new .sxm has been found
            elif isinstance(obj, STS):
                spectra.append(obj)

    return files


def create_map(files, srange=True, smin=20e-9, cmap="afmhot", **kwargs):
    """

    Creates a map of all .sxm images and .dat spectra contained in files.
    Useful for overall experimental picture.

    Parameters
    ----------
    files: array-like of str
        .sxm or .dat files to be plotted.
    srange: bool
        If True then the plot range is limited to the spectra.
        If False then the plot range encompasses everything in files.
    smin: float
        Smallest plot range in x and y, default is 20e-9 (20 nm).
    cmap: str
        Colormap for images.
    kwargs: passed to plt.subplots.
    
    Returns
    -------
    fig: plt.figure
        Figure with plotted and formatted axes.

    """

    kwargs.setdefault("figsize", 2 * _plt.figaspect(1))
    kwargs.setdefault("dpi", 200)

    fig, ax = _plt.subplots(**kwargs)

    # get xy range of all images and spectra
    imranges = []
    sranges = []

    for f in files:
        if f.endswith(".sxm"):
            # open file
            sxm = SXM(f)

            # get correct location from SCAN_RANGE and SCAN_OFFSET
            extent = [
                sxm.header["SCAN_OFFSET"][0]
                - sxm.header["SCAN_RANGE"][0] / 2.0,  # xmin
                sxm.header["SCAN_OFFSET"][0]
                + sxm.header["SCAN_RANGE"][0] / 2.0,  # xmax
                sxm.header["SCAN_OFFSET"][1]
                + sxm.header["SCAN_RANGE"][1] / 2.0,  # ymax
                sxm.header["SCAN_OFFSET"][1]
                - sxm.header["SCAN_RANGE"][1] / 2.0,  # ymin
            ]

            # set zorder to be inversely proportional to area (in nm)
            # -> smaller images on top
            # -> values should all be less than 1 so points are on top (zorder = 2)
            zorder = _constants.nano ** 2 / (
                sxm.header["SCAN_RANGE"][0] * sxm.header["SCAN_RANGE"][1]
            )
            # do plotting
            ax.matshow(sxm.data[0], extent=extent, cmap=cmap, zorder=zorder)

            # get image ranges to sort axes limits later
            imranges.append(extent)

        else:
            # open spectra
            sts = STS(f)
            # plot
            ax.plot(
                sts.header["X (m)"],
                sts.header["Y (m)"],
                marker="o",
                mfc="w",
                mec="k",
                ls="None",
            )
            # get range for sorting axes limits
            sranges.append((sts.header["X (m)"], sts.header["Y (m)"]))

    # confine to spectral range
    if sranges:
        xmin = min([i[0] for i in sranges])
        xmax = max([i[0] for i in sranges])
        ymin = min([i[1] for i in sranges])
        ymax = max([i[1] for i in sranges])

        xrange = abs(xmax - xmin)
        yrange = abs(ymax - ymin)

        # if spectral range is smaller than smin then pad
        if xrange < smin:
            xmin -= (smin - xrange) / 2.0
            xmax += (smin - xrange) / 2.0
        if yrange < smin:
            ymin -= (smin - yrange) / 2.0
            ymax += (smin - yrange) / 2.0

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)

    # otherwise use ranges of all files in list
    else:
        xmin = min([i[0] for i in imranges])
        xmax = max([i[1] for i in imranges])
        ymax = max([i[2] for i in imranges])
        ymin = max([i[3] for i in imranges])

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)

    # sort out axes area with ticks and labels etc.
    ax.set_aspect("equal")
    ax.set_xticklabels(_np.round(ax.get_xticks() / _constants.nano, decimals=1))
    ax.set_yticklabels(_np.round(ax.get_yticks() / _constants.nano, decimals=1))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.xaxis.set_label_position("top")

    fig.tight_layout()

    return fig
