# Module and library calls
import os, csv, requests, logging, argparse, sys, matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict

matplotlib.use('TkAgg')

# Setting up logging
# Setting default to DEBUG in root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Make sure that all messages will be processed by the logger

# Setting up log file handler
file_handler = logging.FileHandler('autompg2.log', 'w')  # Overwrite existing content
file_handler.setLevel(logging.DEBUG)  # Print to log at DEBUG level
logger.addHandler(file_handler)  # Add handler

# Setting up stream/console handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Print to console at INFO level
logger.addHandler(stream_handler)  # Add handler


class AutoMPG:
    def __init__(self, make: str, model: str, year: int, mpg: float):  # Initializing attributes
        """Class constructor. Also initializing attributes to access instance data"""
        self.make = make  # Manufacturer - first token in the 'car name' field
        self.model = model  # Model - all the other tokens in the 'car name' field
        self.year = 1900 + year  # Year of manufacture - four-digit 'model year' field
        self.mpg = mpg  # Miles per gallon - floating point 'mpg' field
        logging.debug('Created AutoMPG-class object of form %r', self)  # Write each object creation to log

    def __repr__(self):
        """Canonical representation of the object in text form"""
        return f'AutoMPG(Make - {self.make} Model - {self.model} Year - {self.year} MPG - {self.mpg})'

    def __str__(self):
        """A more verbose, comprehensible representation of the object in text form"""
        return (
            f'Automobile manufacturer {self.make} manufactured this model, the {self.model}, in {self.year}.'
            f' It was observed to get {self.mpg} miles per gallon.'
        )

    def __eq__(self, other):
        """Implementing equality comparison between two AutoMPG objects"""
        if type(self) is type(other):  # If the two objects are of the same data type
            logging.debug('Comparing two AutoMPG objects: %r == %r', self, other)  # Write comparisons to log
            return (
                    self.make == other.make and
                    self.model == other.model and
                    self.year == other.year and
                    self.mpg == other.mpg
            )
        else:
            return NotImplemented  # Allowing for external overwrites if available

    def __lt__(self, other):
        """Implementing less-than comparison between two AutoMPG objects, using all four attributes in above order"""
        if type(self) is type(other):  # If the two objects are of the same data type
            logging.debug('Comparing two AutoMPG objects: %r < %r', self, other)  # Write comparisons to log
            # This needs to preserve the oder of comparison defined above
            return ((self.make,
                     self.model,
                     self.year,
                     self.mpg)
                    <
                    (other.make,
                     other.model,
                     other.year,
                     other.mpg))
        else:
            return NotImplemented

    def __hash__(self):
        """Ensuring the class is hashable"""
        # Per convention, placing the class attributes into a tuple and then hashing the tuple
        return hash((self.make, self.model, self.year, self.mpg))


# Now, setting up the separate AutoMPGData class
Record = namedtuple('Record',
                    ['mpg',
                     'cylinders',
                     'displacement',
                     'horsepower',
                     'weight',
                     'acceleration',
                     'model_year',
                     'origin',
                     'car_name',
                     ])


class AutoMPGData:
    def __init__(self):  # No arguments needed
        self.data = []  # Instantiating an empty list to dump the AutoMPG objects into
        logging.debug('Created AutoMPGData-class object')  # Write each object's creation to the log file
        self.load_data()  # Call the load_data method

    def __iter__(self):  # Making the class iterable
        return iter(self.data)  # Lists are iterable, so we don't need __next__

    def load_data(self):
        # It makes the most sense to fix data errors at the loading stage
        # I've decided to write a dictionary with the specified corrections and call .get() on it
        # This is because .get() will return the original if there's no correction
        corrected_makes = {
            'chevroelt': 'chevrolet',
            'chevy': 'chevrolet',
            'maxda': 'mazda',
            'mercedes-benz': 'mercedes',
            'toyouta': 'toyota',
            'vokswagen': 'volkswagen',
            'vw': 'volkswagen'
        }

        if not os.path.exists('autompg.data.txt'):  # If the raw data file is not present
            # Send a warning to the console
            logging.warning('No data file exists. Attempting to retrieve from web - calling get_data method.')
            self.get_data()  # Then run the get_data method

        if not os.path.exists('autompgdata.clean.txt'):  # If a clean file is not present
            logging.info('No clean data file exists. Running clean_data().')  # Send a notification to the console
            self.clean_data()  # And run the clean_data method

        with open('autompgdata.clean.txt', 'r') as file:  # Open the clean data file
            text_data = csv.reader(file, delimiter=' ', skipinitialspace=True)  # Delimit on spaces, skip multiples

            for line in text_data:  # For each row in the data
                record = Record(*line)  # Create a 'Record' namedtuple object with the clean_line elements

                car_name_list = record.car_name.split(' ', 1)  # Split the car_name field into make and model

                if len(car_name_list) < 2:  # If there's no model
                    car_name_list.append('')  # Append an empty character, so we don't mess up the make/model split

                # Now we can call the .get() method to correct the specified car makes
                corrected_make = corrected_makes.get(car_name_list[0], car_name_list[0])

                # Create the AutoMPG objects, referencing the corrected_makes dictionary for make
                auto_mpg = AutoMPG(corrected_make, car_name_list[1], int(record.model_year), float(record.mpg))
                # And append them to the list
                self.data.append(auto_mpg)

                logging.debug('AutoMPG object added to list: %r', auto_mpg)  # Write all additions to the log

        # Ending output with count of objects created and added to the list printed to console using logging
        logging.info(f'List created. Total objects: {len(self.data)}')  # Log data loading completion with INFO level

    def clean_data(self):
        logging.info('clean_data method is running. Creating data file now.')  # Send notification to the console
        with open('autompg.data.txt', 'r') as file_input:  # Opening raw file in read mode
            with open('autompgdata.clean.txt', 'w') as file_output:  # And also creating a file for the cleaned data

                for line in file_input:  # For each line in the raw file
                    clean_line = line.expandtabs(1)  # Replace the tab character with a single space
                    file_output.write(clean_line)  # Write each clean line to the new file

    # Creating sort_by_default method
    def sort_by_default(self):
        logging.info('Sorting data list using default order - make, model, year, mpg.')  # Send notification to console
        self.data.sort()  # Sorting using default order defined in the autoMPG class' __lt__ method

    # Creating sort_by_year method
    def sort_by_year(self):
        logging.info('Sorting data list using year first - year, make, model, mpg.')  # Send notification to console
        self.data.sort(key=lambda order: (order.year, order.make, order.model, order.mpg))

    # Creating sort_by_mpg method
    def sort_by_mpg(self):
        logging.info('Sorting data list using mpg first - mpg, make, model, year.')  # Send notification to console
        self.data.sort(key=lambda order: (order.mpg, order.make, order.model, order.year))

    # Creating get_data method using requests module
    def get_data(self):
        web_data = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')

        if not web_data.raise_for_status():  # If an error is not raised
            with open('autompg.data.txt', 'w') as data_file:  # Open a file path in write mode
                data_file.write(web_data.text)  # And write the file as text
            logging.info('Data file fetched from web and saved to disk.')  # Send notification to console

        else:  # If we get an error retrieving the data from the web
            logging.error('ERROR: Unable to fetch data file from web!')  # Send error message to console

    # Creating the mpg_by_year method
    def mpg_by_year(self):
        mpg_by_year_dict = defaultdict(list)  # Instantiating a defaultdict defaulting to an empty list for a given year

        for line in self.data:  # For each line in the AutoMPGData class' data
            mpg_by_year_dict[line.year].append(line.mpg)  # Append each line's mpg to the corresponding model year

        for year in mpg_by_year_dict.keys():  # For each 'year' key in the dictionary
            # Compute the average at the year level by dividing sum of mpg entries by length at the year level
            mpg_by_year_dict[year] = sum(mpg_by_year_dict[year]) / len(mpg_by_year_dict[year])

        # Finally, return a sorted dictionary, which is needed for the CLI functionality
        return dict(sorted(mpg_by_year_dict.items()))  # We can sort by items, which represent the k-v pairs

    # Creating the mpg_by_make method
    def mpg_by_make(self):
        mpg_by_make_dict = defaultdict(list)  # Instantiating a defaultdict defaulting to an empty list for a given make

        for line in self.data:  # For each line in the AutoMPGData class' data
            mpg_by_make_dict[line.make].append(line.mpg)  # Append each line's mpg to the corresponding model year

        for make in mpg_by_make_dict.keys():  # For each 'make' key in the dictionary
            # Compute the average at the make level by dividing sum of mpg entries by length at the make level
            mpg_by_make_dict[make] = sum(mpg_by_make_dict[make]) / len(mpg_by_make_dict[make])

        # Finally, return a sorted dictionary, which is needed for the CLI functionality
        return dict(sorted(mpg_by_make_dict.items()))  # We can sort by items, which represent the k-v pairs


def main():
    autompg_data = AutoMPGData()

    # Setting up the CLI functionalities
    if command == 'print':  # When the print command is entered:

        # Check for each sort option
        if sort_order == 'year':
            autompg_data.sort_by_year()

        elif sort_order == 'mpg':
            autompg_data.sort_by_mpg()

        # And default to default
        else:
            autompg_data.sort_by_default()

        # Opening the csv writer from the csv module
        writer = csv.writer(outfile)  # Writing to the outfile if one is specified by the user
        writer.writerow(['Make', 'Model', 'Year', 'MPG'])  # Providing a header row

        # Now we can print the records
        for record in autompg_data:
            writer.writerow([record.make, record.model, record.year, record.mpg])  # Write each row to csv
            # Should still print normally to console if no outfile is specified because of how argparse.FileType() works

    elif command == 'mpg_by_year':  # If a user calls the mpg_by_year method
        year_mpg_dict = autompg_data.mpg_by_year()  # Ensuring we call the correct return value, a dictionary

        writer = csv.writer(outfile)  # Opening the csv writer from the csv module
        writer.writerow(['Year', 'Average MPG'])  # Providing a header row

        for year, mpg in year_mpg_dict.items():  # Note that average mpg was already computed as the item for each year
            writer.writerow([year, mpg])  # Writing aggregated data

        # If user passes the plot argument, make the plot
        if make_plot:
            plt.plot(list(year_mpg_dict.keys()), list(year_mpg_dict.values()))
            plt.xlabel('Year')
            plt.ylabel('Average MPG')
            plt.title('Average MPG by Year')
            plt.show()

    elif command == 'mpg_by_make':  # If a user calls the mpg_by_make method
        make_mpg_dict = autompg_data.mpg_by_make()  # Ensuring we call the correct return value, a dictionary

        writer = csv.writer(outfile)  # Opening the csv writer from the csv module
        writer.writerow(['Make', 'Average MPG'])  # Providing a header row

        for make, mpg in make_mpg_dict.items():  # Note that average mpg was already computed as the item for each make
            writer.writerow([make, mpg])  # Writing aggregated data

        if make_plot:
            plt.plot(list(make_mpg_dict.keys()), list(make_mpg_dict.values()))
            plt.xlabel('Make')
            plt.ylabel('Average MPG')
            plt.title('Average MPG by Make')
            plt.show()


if __name__ == '__main__':
    # Setting up the argument parser
    parser = argparse.ArgumentParser(description='Analyze Auto MPG data set')

    # Required print command - updated to support additional commands
    parser.add_argument('command', metavar='<command>', type=str, choices=['print', 'mpg_by_year', 'mpg_by_make'],
                        help='Options are print, mpg_by_year, or mpg_by_make')

    # Optional sort command
    parser.add_argument('-s', '--sort', metavar='<sort order>', type=str, choices=['year', 'mpg', 'default'],
                        default='default')

    # Optional outfile command
    # Requires the sys library to default to stdout
    parser.add_argument('-o', '--ofile', metavar='<outfile>', type=argparse.FileType('w'), default=sys.stdout,
                        help='Provide a filepath for non-console output')

    # Optional 'make a plot' command
    parser.add_argument('-p', '--plot', metavar='<makeplot>',
                        help='Create a plot (Usable with mpg_by_year and mpg_by_make commands only)')

    # Storing the parsed arguments
    arguments = parser.parse_args()

    # And we need these to pass into the main function
    command = arguments.command
    sort_order = arguments.sort
    outfile = arguments.ofile  # Holds the name/path of the file provided if the user specifies one on the CLI
    make_plot = arguments.plot  # Holds the command to construct a plot

    main()
