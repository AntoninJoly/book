import pandas as pd


class PreProcess(object):

    def __init__(self, df):
        self.df = df
        self.dropped_columns_ = ['_id', 'city', 'latitude', 'longitude', 'state', 
                                'street', 'zipcode', 'appliances', 'architecture', 
                                'basement', 'coolingsystem', 'exteriormaterial', 
                                'floorcovering', 'floornumber', 'heatingsources', 
                                'heatingsystem', 'numfloors', 'numunits', 'roof', 
                                'rooms', 'view', 'yearupdated', 'elementaryschool', 
                                'highschool', 'homedescription', 'images_count', 
                                'images_image', 'links_homedetails', 'links_homeinfo', 
                                'links_photogallery', 'middleschool', 'neighborhood', 
                                'pageviewcount_currentmonth', 'pageviewcount_total', 
                                'agentname' ,'agentprofileurl', 'externalurl', 
                                'lastupdateddate', 'mls', 'openhousedates', 'status', 
                                'schooldistrict', 'usecode', 'type', 'trans_desc', 
                                'trans_summary', 'walkscore_desc', 'whatownerloves', 'zpid']

    def drop_columns(self):
        """
        Notes: this function will drop all of the columns from a dataframe that are not
        needed for computation of any of the similarity metrics. For example, 'heatingsources'
        is not currently used in any of the similarity metrics, so it is dropped.
        """
        self.df = self.df.drop(self.dropped_columns_, axis=1)
        try:
            self.df = self.df.drop('editedfacts', axis=1)
        except:
            pass
        return self.df
    
    def preprocess_df(self,df):
        df = df.drop(['numrooms', 'price'], axis=1) #there are too many NA values for price for it ot be useful
        df = df.dropna(axis=0, how='any')
        
        # remove 'None' values in trans_score
        indices = df[df['trans_score'].isin(['None'])].index
        df = df[~(df.index.isin(indices))]
        indices = df[df['walkscore_score'].isin(['None'])].index
        df = df[~(df.index.isin(indices))]
        
        # cast as ints
        df[['trans_score', 'walkscore_score']] = df[['trans_score', 'walkscore_score']].astype(int) 
        
        # remove the rows that are absurdly big and are probably mistakes
        df = df[df.bedrooms <= 20]
        df = df[df.bathrooms <= 20]
        df[['bathrooms', 'bedrooms', 'finishedsqft']] = df[['bathrooms', 'bedrooms', 'finishedsqft']].astype(float)
        return df
    
    def normalize_num(self, x, col_min, col_max):
        """
        Normalize everything from 0 to 1 
        """
        return float((x - col_min)) / (col_max - col_min)
    
    def normalize_columns(self,df):
        """
        Input: a list of columns to normalize
        Output: a dataframe that has normalized the columns between 0 and 1
        """
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = df[col].apply(self.normalize_num, args=(min_val, max_val))
        return df

    def create_parking_index(self,df):
        """
        Input: a dataframe
        Output: a dateframe
        Notes: this function will take the uncleaned columns of parkingtype and coveredparkingspaces and
        apply score_parking to return a column that is a numerical index of parking quality for a listing
        """
        cols = ['parkingtype', 'coveredparkingspaces']
        df[cols] = df[cols].fillna('None')
        
        # score_parking is created and called within the scope of create_parking_index
        def score_parking(parkingtype, coveredparkingspaces):
            type_score = 0
            space_score = 0
            if parkingtype != "None":
                if "Garage - Attached" in parkingtype:
                    type_score += 0.5
                elif "Garage - Detached" in parkingtype:
                    type_score += 0.4
                elif "Carport" in parkingtype:
                    type_score += 0.3
                elif "Off-street" in parkingtype:
                    type_score += 0.2
                elif "On-street" in parkingtype:
                    type_score += 0.1
            else:
                type_score = 0
            if coveredparkingspaces != "None":
                if coveredparkingspaces >= 5:
                    space_score += 0.5
                else:
                    space_score += coveredparkingspaces * 0.1
            else:
                space_score = 0
            return type_score + space_score
        
        df['parking_index'] = df[cols].apply(lambda x: score_parking(x[cols[0]], x[cols[1]]), axis=1)
        df = df.drop(cols, axis=1) # drop the unnecessary columns
        return df

    def filter_df(self, metric):
        """
        Input: takes a dataframe, and the name of a similarity metric as a string
        Output: the cleaned and filtered df for the appropraite metric

        Notes: this function is meant to trim the dataframe to use only the 
        columns that are relevant to a given similarity metric.  
        """
        if metric == "walk_distance":
            self.df = self.df[['trans_score', 'walkscore_score']].dropna(axis=0, how='any')
            return self.df.astype(float)
        if metric == "house_superficy":
            self.df = self.df[['bathrooms', 'bedrooms', 'finishedsqft']].dropna(axis=0, how='any')
            return self.df.astype(float)
        if metric == "family_amenities":
            self.df = self.df[['bedrooms', 'yearbuilt', 'lotsizesqft', 'parking_index']]
            return self.df



