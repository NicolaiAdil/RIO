import rospy

from numpy import deg2rad

NED_munkholmen = rospy.get_param("/autosea/ned/munkholmen/id")
NED_munkholmen_llh = rospy.get_param("/autosea/ned/munkholmen/llh")
NED_dorabassenget_llh = rospy.get_param("/autosea/ned/dorabassenget/llh")
NED_brattorabassenget_llh = rospy.get_param("/autosea/ned/brattorabassenget/llh")
world_seapath = rospy.get_param("/autosea/seapath/world")
NED_seapath = rospy.get_param("/autosea/seapath/ned")
body_seapath = rospy.get_param("/autosea/seapath/body")
surface_seapath = rospy.get_param("/autosea/seapath/surface")

lat_min, lat_max, lon_min, lon_max = "lat_min", "lat_max", "lon_min", "lon_max"
munkholmen_lla = {
    lat_min: deg2rad(rospy.get_param("/autosea/munkholmen_lla/lat/min")),
    lat_max: deg2rad(rospy.get_param("/autosea/munkholmen_lla/lat/max")),
    lon_min: deg2rad(rospy.get_param("/autosea/munkholmen_lla/lon/min")),
    lon_max: deg2rad(rospy.get_param("/autosea/munkholmen_lla/lon/max")),
}
pier_lla = {
    lat_min: deg2rad(rospy.get_param("/autosea/pier_lla/lat/min")),
    lat_max: deg2rad(rospy.get_param("/autosea/pier_lla/lat/max")),
    lon_min: deg2rad(rospy.get_param("/autosea/pier_lla/lon/min")),
    lon_max: deg2rad(rospy.get_param("/autosea/pier_lla/lon/max")),
}
trondheimsfjord_lla = {
    lat_min: deg2rad(rospy.get_param("/autosea/trondheimsfjord_lla/lat/min")),
    lat_max: deg2rad(rospy.get_param("/autosea/trondheimsfjord_lla/lat/max")),
    lon_min: deg2rad(rospy.get_param("/autosea/trondheimsfjord_lla/lon/min")),
    lon_max: deg2rad(rospy.get_param("/autosea/trondheimsfjord_lla/lon/max")),
}
gunnerus_lla = {
    lat_min: deg2rad(rospy.get_param("/autosea/gunnerus_lla/lat/min")),
    lat_max: deg2rad(rospy.get_param("/autosea/gunnerus_lla/lat/max")),
    lon_min: deg2rad(rospy.get_param("/autosea/gunnerus_lla/lon/min")),
    lon_max: deg2rad(rospy.get_param("/autosea/gunnerus_lla/lon/max")),
}
