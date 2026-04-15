class GetAttriAble:
    @classmethod
    def __get_attribute__(cls):
        # Getting attribute (variable) of this class
        result = []
        for item in cls.__dict__:
            if "__" in item: # if 
                continue
            result.append(item)
        return result
    
    def __setitem__(self, key, value):
        if key not in self.__get_attribute__():
            raise KeyError(f"{key} is not a valid attribute of {self.__class__.__name__}")
        setattr(self, key, value)
        
class SearchResult(GetAttriAble):
    Id = None
    Name = None
    Address = None
    District = None 
    City = None
    Phone = None
    AvgRatingOriginal = None
    Cuisines: list | None = None
    AlbumUrl = None
    HasDelivery = None
    HasBooking = None
    BookingUrl = None
    HasVideo = None
    Services: list | None = None
    Categories: list | None = None
    DetailUrl = None
    BranchUrl = None
    TotalReviews = None
    TotalView = None 
    TotalFavourite = None
    TotalCheckins = None
    Status = None
    review_url = None
    PicturePath = None
    PicturePathLarge = None
    MobilePicturePath = None
    SubItems = None
    MainCategoryId = None

    

class BranchResult(GetAttriAble):
    Id = None
    Name = None
    Address = None
    AvgRating = None
    TotalPictures = None 
    Url: str | None = None
    HasPromotion = None
    HasMemberCard = None
    MemberCardDiscount = None
    Latitude = None
    Longitude = None
    ResCreatedOn = None
    PriceMin = None
    PriceMax = None
    District = None
    ResUrlAlbums = None 
    ResUrlReviews = None
    Services: list | None = None
    UrlRewriteName = None
    RestaurantId = None
    MainCategoryId = None
    HasVideo = None
    HasBooking = None
    IsBooking = None
    HasDelivery = None
    IsDelivery = None
    BookinngUrl = None
    DeliveryUrl = None
    Latitude = None
    Longitude = None
    FTotalReviews = None
    FAvgRating = None
    LocationUrlRewriteName = None

    
class ReviewResult(GetAttriAble):
    ResId = None
    Id = None
    Title = None
    Desciption = None
    IsAllowComment = None
    AvgRating = None
    Video = None
    Picture = None
        # Width
        # Heigth
        # PhotoDetailUrl
        # Url
    Owner = None 
        # TrustPercent
        # VeriyingPercent 
    Options = None
        # VisitAgain
        # MoneySpend

class StoreDetails(GetAttriAble):
    PictureModel: dict | tuple | None = None
        # ImageUrl
        # Title
    HasValidContract = None
    TableNowApiEnable = None
    RestaurantID = None
    Name = None
    Address = None 
    District = None
    Area = None
    AreaId = None
    City = None
    CityId = None
    PriceMin = None
    PriceMax = None
    IsActive = None
    MetaTitle = None
    MetaKeywords = None
    MetaDescription = None
    AvgPointList: list | None = None
        # Label
        # Point
    Cuisines = None
    LstTargetAudience: list | None = None
        # Name
    LstCategory: list | None = None
        # AsciiName 
        # Name
        # Id
    Services: list | None = None
    AccessGuide = None
    DeliveryId = None
    Ratings:dict | None = None
    RestaurantUrl = None
    ResUrlRewrite = None
    Properties: list | None = None

  
if __name__ == "__main__":
    print(BranchResult.__get_attribute__())
    print(SearchResult.__get_attribute__())
    
